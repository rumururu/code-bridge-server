# RevenueCat → Supabase Webhook Setup

Owner: Code Bridge backend.
Audit reference: `docs/audit_findings/2026-06-05_full_stack_audit.md` SUB-3.

This runbook wires RevenueCat subscriber events into the Code Bridge
headless server so the server has a **canonical, push-based view** of
entitlement state instead of only the 5-minute REST cache from
RevenueCat. The path is:

```
RevenueCat dashboard
   ──webhook POST──►  Supabase Edge Function: revenuecat-webhook
                     ─upsert─►  public.entitlement_state
                                       ▲
                                       │ REST (fast-path)
                                       │
                       Code Bridge headless server
                       (entitlement_service.py)
```

If the Supabase fast-path is unavailable, the server falls back to the
existing RevenueCat REST lookup → stale cache → anonymous denial, in
that order.

---

## Priority order in `entitlement_service.py`

After this runbook is followed, `EntitlementService.is_entitled` resolves
in this order:

1. **Local override** (`CODE_BRIDGE_ENTITLEMENT_OVERRIDE=active`) — host
   developer / self-hosted operator.
2. **Supabase `entitlement_state` row** — populated by this webhook.
   Source: `"supabase"`.
3. **RevenueCat REST** (`GET /v1/subscribers/{app_user_id}`) — source:
   `"revenuecat"`.
4. **Stale Supabase or RevenueCat cache** — entries up to 1 hour old,
   served with a `WARNING` log if upstream is down.
5. **`anonymous` / `no_subscription` denial.**

---

## Step 1 — Supabase project

Code Bridge re-uses the `mkideabox` Supabase organisation. Either:

- pick the existing `mkideabox-prod` project and add the table there
  (recommended — one billing surface), or
- create a new project `code-bridge-prod`. If you do, set its
  `Project URL` and `service_role` key aside; you will need them in
  step 4.

You need the [Supabase CLI](https://supabase.com/docs/guides/cli)
installed locally:

```bash
brew install supabase/tap/supabase
supabase --version   # ≥ 1.150
```

Link your local repo to the project once:

```bash
cd /Users/mankil/AndroidStudioProjects/code_bridge
supabase link --project-ref <project-ref>
```

(`<project-ref>` is the 20-character slug in your project URL, e.g.
`abcdefghijklmnopqrst`.)

## Step 2 — Run the SQL migration

The migration lives at
`supabase/migrations/20260605070651__entitlement_state.sql` and creates:

- `public.entitlement_state` (one row per subscriber, RLS enabled — only
  the service_role and the matching `auth.uid()` can read it)
- `public.processed_events` (idempotency ledger, service_role only)

Apply it:

```bash
supabase db push
```

Verify in the SQL editor:

```sql
select * from public.entitlement_state limit 1;
select * from public.processed_events limit 1;
```

Both should exist and be empty.

## Step 3 — Set the webhook shared secret

RevenueCat sends a shared secret in the `Authorization: Bearer …`
header, NOT an HMAC signature. Generate one:

```bash
openssl rand -hex 32
```

Store it in 1Password under "RevenueCat → Code Bridge webhook secret",
then set it on the Edge Function:

```bash
supabase secrets set REVENUECAT_WEBHOOK_SECRET=<the-hex-string>
```

`SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` are auto-injected by
Supabase Functions — you do not need to set them yourself.

## Step 4 — Deploy the function

```bash
supabase functions deploy revenuecat-webhook
```

The function URL will be printed at the end; it looks like:

```
https://<project-ref>.supabase.co/functions/v1/revenuecat-webhook
```

Save it for the next step.

## Step 5 — Configure RevenueCat dashboard

1. Open `https://app.revenuecat.com` → your Code Bridge project →
   **Integrations** → **Webhooks**.
2. Click **Add Webhook**.
3. **URL**: paste the function URL from step 4.
4. **Authorization header**: paste `Bearer <secret>` where `<secret>`
   is the value from step 3.
5. Select event types: at minimum tick `INITIAL_PURCHASE`, `RENEWAL`,
   `CANCELLATION`, `EXPIRATION`, `BILLING_ISSUE`, `PRODUCT_CHANGE`,
   `TRIAL_STARTED`, `TRIAL_CONVERTED`, `NON_RENEWING_PURCHASE`,
   `UNCANCELLATION`. Other types are tolerated (200, status=ignored).
6. Click **Send Test Event** — the dashboard should show a `200 OK`
   response within ~1 s.

## Step 6 — Point the Code Bridge server at Supabase

On every machine that runs the headless server (`server/`), set:

```bash
export SUPABASE_URL="https://<project-ref>.supabase.co"
export SUPABASE_SERVICE_ROLE_KEY="<service-role-key>"
```

For systemd installs this goes in
`/etc/systemd/system/code-bridge.service.d/override.conf`:

```ini
[Service]
Environment=SUPABASE_URL=https://<project-ref>.supabase.co
Environment=SUPABASE_SERVICE_ROLE_KEY=<service-role-key>
```

Reload + restart:

```bash
sudo systemctl daemon-reload
sudo systemctl restart code-bridge
```

For Docker installs add the same two env vars to your `docker run` or
`docker-compose.yml`.

> **Why service_role and not anon?** The headless server queries
> `entitlement_state` for arbitrary `app_user_id` values, so RLS would
> block the `anon` key. The service_role key bypasses RLS and is safe
> here because the server already protects the key the same way it
> protects `REVENUECAT_SECRET_API_KEY`.

## Step 7 — Verification

### 7a. End-to-end with a sandbox purchase

1. Sign into Code Bridge on a sandbox-enabled iOS or Android device.
2. Purchase the `$0.99/month` subscription via the in-app paywall.
3. Within ~5 s, query Supabase:

   ```sql
   select * from public.entitlement_state where app_user_id = '<your-uid>';
   ```

   You should see `active = true`, `last_event = 'INITIAL_PURCHASE'`,
   `expires_at` set to ~1 month out.
4. From the headless server logs, the next entitlement check for that
   user should log `source=supabase` (DEBUG level).

### 7b. Cancellation drill

1. In the RevenueCat dashboard, find the subscriber and **revoke**
   their entitlement.
2. Within ~5 s, the same row should flip to `active = false`,
   `last_event = 'CANCELLATION'`.
3. The server's 5-minute cache for that user will keep returning
   active until it next refreshes — that is by design (we trade a few
   minutes of staleness for fewer Supabase round-trips). To force a
   refresh, restart the server.

### 7c. Idempotency

Trigger the same test event twice from the RevenueCat dashboard. The
second call should return `200 {"status":"duplicate"}` and the row's
`updated_at` should not change.

### 7d. Failure modes

- Wrong secret → function returns `401 unauthorized`. Confirm by
  hitting it with `curl -H 'Authorization: Bearer wrong' …`.
- Supabase down → the headless server logs
  `entitlement: Supabase fast-path failed, falling back` (WARNING) and
  serves the RevenueCat REST result. Verified by the unit test
  `test_supabase_5xx_falls_through_to_revenuecat`.

---

## Rollback

If the webhook misbehaves:

1. **Disable** it in the RevenueCat dashboard (do not delete — keeps
   the configured URL/secret for re-enabling).
2. Unset `SUPABASE_URL` / `SUPABASE_SERVICE_ROLE_KEY` on the server and
   restart. The server will revert to RevenueCat REST → cache →
   anonymous, matching pre-SUB-3 behaviour.
3. Investigate Supabase Function logs:
   `supabase functions logs revenuecat-webhook --tail`.

## Cost

Supabase Edge Functions: 500K invocations/month free, $2/M after. At
~10K subscribers churning monthly that is well under the free tier.
