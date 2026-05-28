#!/usr/bin/env python3
"""Create and import an Apple Developer ID Application certificate.

This uses the App Store Connect API directly because some fastlane versions map
Developer ID Application to an API enum that Apple no longer accepts.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import plistlib
import secrets
import string
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import jwt


API_BASE = "https://api.appstoreconnect.apple.com/v1"


def run(command: list[str], *, quiet: bool = False) -> subprocess.CompletedProcess[str]:
    if not quiet:
        print("$ " + " ".join(command))
    return subprocess.run(command, check=True, text=True, capture_output=quiet)


def load_api_key(path: Path) -> tuple[str, str, str]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    key_id = payload["key_id"]
    issuer_id = payload["issuer_id"]
    key = payload["key"]
    return key_id, issuer_id, key


def make_token(key_id: str, issuer_id: str, private_key: str) -> str:
    now = int(time.time())
    return jwt.encode(
        {
            "iss": issuer_id,
            "iat": now,
            "exp": now + 20 * 60,
            "aud": "appstoreconnect-v1",
        },
        private_key,
        algorithm="ES256",
        headers={"kid": key_id, "typ": "JWT"},
    )


def request_json(token: str, method: str, path: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        API_BASE + path,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Apple API request failed: HTTP {exc.code}\n{detail}") from exc


def keychain_path() -> str:
    result = run(["security", "default-keychain"], quiet=True)
    raw = result.stdout.strip()
    return raw.strip('"')


def has_developer_id_identity() -> bool:
    result = subprocess.run(
        ["security", "find-identity", "-v", "-p", "codesigning"],
        text=True,
        capture_output=True,
        check=False,
    )
    return "Developer ID Application:" in result.stdout


def create_csr(work_dir: Path, common_name: str) -> tuple[Path, Path]:
    private_key = work_dir / "developer_id_application.key"
    csr = work_dir / "developer_id_application.csr"
    run(
        [
            "openssl",
            "req",
            "-new",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(private_key),
            "-out",
            str(csr),
            "-subj",
            f"/CN={common_name}",
        ],
    )
    return private_key, csr


def create_certificate(token: str, csr_path: Path) -> bytes:
    csr_content = csr_path.read_text(encoding="utf-8")
    response = request_json(
        token,
        "POST",
        "/certificates",
        {
            "data": {
                "type": "certificates",
                "attributes": {
                    "certificateType": "DEVELOPER_ID_APPLICATION",
                    "csrContent": csr_content,
                },
            },
        },
    )
    encoded = response["data"]["attributes"]["certificateContent"]
    return base64.b64decode(encoded)


def import_certificate(private_key: Path, cert_der: bytes, common_name: str) -> None:
    with tempfile.TemporaryDirectory(prefix="codebridge-developer-id-import-") as tmp:
        tmp_dir = Path(tmp)
        cert_path = tmp_dir / "developer_id_application.cer"
        pem_path = tmp_dir / "developer_id_application.pem"
        p12_path = tmp_dir / "developer_id_application.p12"
        password = "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(32))

        cert_path.write_bytes(cert_der)
        run(["openssl", "x509", "-inform", "DER", "-in", str(cert_path), "-out", str(pem_path)])
        run(
            [
                "openssl",
                "pkcs12",
                "-export",
                "-legacy",
                "-inkey",
                str(private_key),
                "-in",
                str(pem_path),
                "-out",
                str(p12_path),
                "-name",
                common_name,
                "-passout",
                f"pass:{password}",
            ],
        )
        run(
            [
                "security",
                "import",
                str(p12_path),
                "-k",
                keychain_path(),
                "-P",
                password,
                "-T",
                "/usr/bin/codesign",
                "-T",
                "/usr/bin/security",
                "-T",
                "/usr/bin/productsign",
            ],
        )


def read_certificate_bytes(path: Path) -> bytes:
    raw = path.read_bytes()
    if b"-----BEGIN CERTIFICATE-----" in raw:
        with tempfile.TemporaryDirectory(prefix="codebridge-developer-id-pem-") as tmp:
            der_path = Path(tmp) / "certificate.der"
            run(["openssl", "x509", "-in", str(path), "-outform", "DER", "-out", str(der_path)])
            return der_path.read_bytes()
    return raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create and import Developer ID Application certificate.")
    parser.add_argument(
        "--api-key-path",
        type=Path,
        default=Path("ios/fastlane/api_key.json"),
        help="fastlane App Store Connect API key JSON.",
    )
    parser.add_argument(
        "--common-name",
        default="Developer ID Application: mkideabox Co. Ltd. (3SAMRT9KZD)",
        help="Common name used for the local CSR/private key label.",
    )
    parser.add_argument("--force", action="store_true", help="Create a new certificate even if one is already installed.")
    parser.add_argument(
        "--write-csr-dir",
        type=Path,
        default=None,
        help="Generate a persistent private key and CSR for manual Apple Developer portal upload.",
    )
    parser.add_argument(
        "--import-cert",
        type=Path,
        default=None,
        help="Import a downloaded Developer ID Application .cer/.pem using --private-key.",
    )
    parser.add_argument(
        "--private-key",
        type=Path,
        default=None,
        help="Private key that was used to create the CSR for --import-cert.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write_csr_dir:
        output_dir = args.write_csr_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        private_key = output_dir / "developer_id_application.key"
        csr = output_dir / "developer_id_application.csr"
        if private_key.exists() or csr.exists():
            raise SystemExit(f"Refusing to overwrite existing CSR material in {output_dir}")
        create_csr(output_dir, args.common_name)
        print(f"CSR written to: {csr}")
        print(f"Private key written to: {private_key}")
        print("Keep the private key private. Upload only the .csr file to Apple Developer.")
        return

    if args.import_cert:
        if not args.private_key:
            raise SystemExit("--private-key is required with --import-cert")
        cert_der = read_certificate_bytes(args.import_cert.resolve())
        import_certificate(args.private_key.resolve(), cert_der, args.common_name)
        if not has_developer_id_identity():
            raise SystemExit("Certificate was imported, but no Developer ID Application signing identity was found.")
        print("Developer ID Application certificate imported successfully.")
        return

    if has_developer_id_identity() and not args.force:
        print("Developer ID Application identity already exists in the keychain.")
        return

    api_key_path = args.api_key_path.resolve()
    key_id, issuer_id, private_key = load_api_key(api_key_path)
    token = make_token(key_id, issuer_id, private_key)

    with tempfile.TemporaryDirectory(prefix="codebridge-developer-id-csr-") as tmp:
        work_dir = Path(tmp)
        key_path, csr_path = create_csr(work_dir, args.common_name)
        cert_der = create_certificate(token, csr_path)
        import_certificate(key_path, cert_der, args.common_name)

    if not has_developer_id_identity():
        raise SystemExit("Certificate was imported, but no Developer ID Application signing identity was found.")
    print("Developer ID Application certificate imported successfully.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        if exc.stderr:
            print(exc.stderr, file=sys.stderr)
        raise SystemExit(exc.returncode) from exc
