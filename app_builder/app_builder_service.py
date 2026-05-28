"""Create local app workspaces and seed Agent Cockpit tasks."""

import json
import re
import shutil
import subprocess
from html import escape
from pathlib import Path
from typing import Any

from agent.agent_store import get_agent_store
from audit.route_audit import record_api_action
from core.base_result import BaseRouteResult
from projects.project_action_service import create_project_folder_for_current_server
from workspaces.workspace_store import get_workspace_store


def create_local_app_for_current_server(
    *,
    root_path: str,
    app_name: str,
    prompt: str,
    template: str = "nextjs",
    provider_id: str | None = None,
    model: str | None = None,
) -> BaseRouteResult:
    """Create a local app folder, project record, workspace, task, and run."""
    normalized_template = _normalize_template(template)
    if normalized_template not in {"nextjs", "vite", "flutter"}:
        return BaseRouteResult.error(400, f"Unsupported app template: {template}")

    clean_name = _project_slug(app_name)
    if not clean_name:
        return BaseRouteResult.error(400, "App name is required")
    if not prompt.strip():
        return BaseRouteResult.error(400, "Prompt is required")

    project_type = _project_type_for_template(normalized_template)
    dev_server = _dev_server_for_template(normalized_template)
    project_result = create_project_folder_for_current_server(
        root_path=root_path,
        folder_name=clean_name,
        requested_name=clean_name,
        requested_type=project_type,
        dev_server=dev_server,
    )
    if not project_result.success:
        return project_result

    project = project_result.payload
    project_path_raw = project.get("path")
    if not isinstance(project_path_raw, str) or not project_path_raw:
        return BaseRouteResult.error(500, "Created project did not include a path")

    project_path = Path(project_path_raw)
    platform_scaffold: dict[str, Any] | None = None
    if normalized_template == "flutter":
        package_name = _dart_package_name(app_name)
        platform_scaffold = _materialize_flutter_platforms(
            project_path,
            package_name=package_name,
        )
        files = _write_flutter_template(
            project_path,
            app_name=app_name,
            prompt=prompt,
            package_name=package_name,
        )
    elif normalized_template == "vite":
        files = _write_vite_template(project_path, app_name=app_name, prompt=prompt)
    else:
        files = _write_nextjs_template(project_path, app_name=app_name, prompt=prompt)

    workspace = get_workspace_store().get_or_create_project_workspace(
        project_name=project["name"],
        root_path=str(project_path),
        display_name=app_name,
        permissions={"roots": [str(project_path)]},
    )

    agent_store = get_agent_store()
    run = agent_store.create_run(
        workspace_id=workspace["id"],
        project_name=project["name"],
        provider_id=provider_id,
        model=model,
        title=f"Create {app_name}",
        goal=prompt,
        cwd=str(project_path),
    )
    agent_store.add_message(
        run_id=run["id"],
        role="user",
        content=prompt,
    )
    tasks = [
        agent_store.create_task(
            workspace_id=workspace["id"],
            run_id=run["id"],
            title=title,
            description=description,
            project_name=project["name"],
            kind="app_build",
            source="app_builder",
            goal=prompt,
            priority=priority,
            labels=["app", normalized_template],
            metadata={
                "template": normalized_template,
                "app_name": app_name,
            },
        )
        for title, description, priority in _initial_task_specs(app_name, prompt)
    ]
    task = tasks[0]
    app_event = {
        "template": normalized_template,
        "project_name": project["name"],
        "workspace_id": workspace["id"],
        "task_ids": [task["id"] for task in tasks],
        "files": files,
    }
    if platform_scaffold is not None:
        app_event["platform_scaffold"] = platform_scaffold
    agent_store.append_event(
        run_id=run["id"],
        event_type="app.created",
        app_event=app_event,
    )
    for file_path in files:
        agent_store.add_artifact(
            run_id=run["id"],
            kind="template_file",
            path=file_path,
            mime_type=_mime_for_path(file_path),
        )
    if platform_scaffold is not None:
        agent_store.add_artifact(
            run_id=run["id"],
            kind="platform_scaffold",
            mime_type="application/json",
            metadata=platform_scaffold,
        )

    record_api_action(
        operation="app.create",
        project_name=project["name"],
        details={
            "root_path": root_path,
            "app_name": app_name,
            "template": normalized_template,
            "file_count": len(files),
            "platform_scaffold": (
                platform_scaffold.get("status")
                if platform_scaffold is not None
                else None
            ),
        },
        success=True,
        status_code=201,
    )

    return BaseRouteResult.ok(
        {
            "project": project,
            "workspace": workspace,
            "task": task,
            "tasks": tasks,
            "run": run,
            "files": files,
            "platform_scaffold": platform_scaffold,
        },
        status_code=201,
    )


def _project_slug(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9_-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-_")
    return text[:64]


def _package_name(value: str) -> str:
    return _project_slug(value) or "code-bridge-app"


def _dart_package_name(value: str) -> str:
    text = _project_slug(value).replace("-", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text or not text[0].isalpha():
        text = f"code_bridge_{text or 'app'}"
    return text[:64]


def _normalize_template(value: str) -> str:
    text = (value or "nextjs").strip().lower()
    if text in {"nextjs", "next", "web"}:
        return "nextjs"
    if text in {"vite", "react", "react-vite", "spa"}:
        return "vite"
    if text in {"flutter", "dart", "mobile"}:
        return "flutter"
    return text


def _project_type_for_template(template: str) -> str:
    if template == "nextjs":
        return "nextjs"
    if template == "vite":
        return "react"
    if template == "flutter":
        return "flutter"
    return "other"


def _dev_server_for_template(template: str) -> dict[str, Any] | None:
    if template == "nextjs":
        return {"command": "npm run dev", "port": 3000}
    if template == "vite":
        return {"command": "npm run dev -- --host 0.0.0.0", "port": 5173}
    return None


def _agent_manifest(app_name: str, prompt: str, template: str, *, dev_port: int) -> dict[str, Any]:
    commands = (
        {
            "install": "flutter pub get",
            "dev": "flutter run",
            "build": "flutter build apk",
            "test": "flutter test",
            "platform_scaffold": "flutter create --platforms=android,ios .",
        }
        if template == "flutter"
        else {
            "install": "npm install",
            "dev": "npm run dev",
            "build": "npm run build",
        }
    )
    payload = {
        "schema": "codebridge.agent.v1",
        "app_name": app_name,
        "initial_prompt": prompt,
        "template": template,
        "commands": commands,
        "acceptance": [
            "Core user workflow is usable from the first screen",
            "Responsive layout works on mobile and desktop",
            f"{commands['build']} completes successfully",
        ],
    }
    if dev_port > 0:
        payload["dev_server"] = {"port": dev_port}
    return payload


def _agent_brief(app_name: str, prompt: str) -> str:
    return (
        f"# Agent Brief: {app_name}\n\n"
        "## Goal\n\n"
        f"{prompt}\n\n"
        "## Delivery checklist\n\n"
        "- Define the primary user workflow.\n"
        "- Replace seed data with real state or API calls.\n"
        "- Verify desktop and mobile layouts.\n"
        "- Attach build, preview, and screenshot results to the Agent Cockpit run.\n"
    )


def _write_nextjs_template(
    project_path: Path,
    *,
    app_name: str,
    prompt: str,
) -> list[str]:
    app_dir = project_path / "app"
    app_dir.mkdir(parents=True, exist_ok=True)

    safe_title = escape(app_name.strip() or "Code Bridge App")
    safe_prompt = escape(prompt.strip())
    agent_manifest = _agent_manifest(app_name, prompt, "nextjs", dev_port=3000)
    package = {
        "name": _package_name(app_name),
        "version": "0.1.0",
        "private": True,
        "scripts": {
            "dev": "next dev -H 0.0.0.0",
            "build": "next build",
            "start": "next start",
            "lint": "next lint",
        },
        "dependencies": {
            "next": "latest",
            "react": "latest",
            "react-dom": "latest",
        },
        "devDependencies": {
            "@types/node": "latest",
            "@types/react": "latest",
            "@types/react-dom": "latest",
            "typescript": "latest",
            "eslint": "latest",
            "eslint-config-next": "latest",
        },
    }
    files = {
        "codebridge.agent.json": json.dumps(agent_manifest, indent=2, ensure_ascii=False) + "\n",
        "package.json": json.dumps(package, indent=2, ensure_ascii=False) + "\n",
        "next.config.mjs": "/** @type {import('next').NextConfig} */\nconst nextConfig = {};\n\nexport default nextConfig;\n",
        "tsconfig.json": json.dumps(
            {
                "compilerOptions": {
                    "target": "es5",
                    "lib": ["dom", "dom.iterable", "esnext"],
                    "allowJs": True,
                    "skipLibCheck": True,
                    "strict": True,
                    "noEmit": True,
                    "esModuleInterop": True,
                    "module": "esnext",
                    "moduleResolution": "bundler",
                    "resolveJsonModule": True,
                    "isolatedModules": True,
                    "jsx": "preserve",
                    "incremental": True,
                    "plugins": [{"name": "next"}],
                },
                "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
                "exclude": ["node_modules"],
            },
            indent=2,
        )
        + "\n",
        ".gitignore": "node_modules\n.next\nout\n.env*.local\n.DS_Store\n",
        "README.md": (
            f"# {app_name}\n\n"
            "Seeded by Code Bridge Agent Cockpit.\n\n"
            "## Initial prompt\n\n"
            f"{prompt}\n\n"
            "## Local commands\n\n"
            "- `npm install`\n"
            "- `npm run dev`\n"
            "- `npm run build`\n"
        ),
        "docs/agent-brief.md": _agent_brief(app_name, prompt),
        "app/layout.tsx": (
            "import './globals.css';\n\n"
            "export const metadata = {\n"
            f"  title: '{_tsx_string(app_name)}',\n"
            "};\n\n"
            "export default function RootLayout({ children }: { children: React.ReactNode }) {\n"
            "  return (\n"
            "    <html lang=\"en\">\n"
            "      <body>{children}</body>\n"
            "    </html>\n"
            "  );\n"
            "}\n"
        ),
        "app/page.tsx": (
            "const workItems = [\n"
            "  { label: 'Product flow', value: 'Draft', tone: 'blue' },\n"
            "  { label: 'Data model', value: 'Open', tone: 'green' },\n"
            "  { label: 'Verification', value: 'Ready', tone: 'amber' },\n"
            "];\n\n"
            "export default function Home() {\n"
            "  return (\n"
            "    <main className=\"appShell\">\n"
            "      <aside className=\"sidebar\">\n"
            f"        <div className=\"brand\">{safe_title}</div>\n"
            "        <nav>\n"
            "          <span className=\"active\">Workspace</span>\n"
            "          <span>Tasks</span>\n"
            "          <span>Preview</span>\n"
            "        </nav>\n"
            "      </aside>\n"
            "      <section className=\"workspace\">\n"
            "        <header className=\"topbar\">\n"
            "          <div>\n"
            "            <p className=\"eyebrow\">Initial agent brief</p>\n"
            f"            <h1>{safe_title}</h1>\n"
            "          </div>\n"
            "          <button>Run build</button>\n"
            "        </header>\n"
            f"        <p className=\"prompt\">{safe_prompt}</p>\n"
            "        <section className=\"statusGrid\">\n"
            "          {workItems.map((item) => (\n"
            "            <article className={`metric ${item.tone}`} key={item.label}>\n"
            "              <span>{item.label}</span>\n"
            "              <strong>{item.value}</strong>\n"
            "            </article>\n"
            "          ))}\n"
            "        </section>\n"
            "        <section className=\"workbench\">\n"
            "          <article>\n"
            "            <h2>Primary workflow</h2>\n"
            "            <p>Replace this panel with the first production workflow for the app.</p>\n"
            "          </article>\n"
            "          <article>\n"
            "            <h2>Agent next steps</h2>\n"
            "            <ol>\n"
            "              <li>Map the core screens and states.</li>\n"
            "              <li>Implement the main interaction path.</li>\n"
            "              <li>Run lint, build, and preview checks.</li>\n"
            "            </ol>\n"
            "          </article>\n"
            "        </section>\n"
            "      </section>\n"
            "    </main>\n"
            "  );\n"
            "}\n"
        ),
        "app/globals.css": (
            "* { box-sizing: border-box; }\n"
            "body { margin: 0; font-family: Arial, Helvetica, sans-serif; background: #f4f7fb; color: #12212f; }\n"
            ".appShell { min-height: 100vh; display: grid; grid-template-columns: 240px 1fr; }\n"
            ".sidebar { background: #12212f; color: #eef6ff; padding: 24px; }\n"
            ".brand { font-weight: 800; font-size: 20px; margin-bottom: 32px; }\n"
            "nav { display: grid; gap: 8px; }\n"
            "nav span { color: #b8c7d9; padding: 10px 12px; border-radius: 8px; }\n"
            "nav .active { background: #1f3a52; color: #ffffff; }\n"
            ".workspace { padding: 32px; display: grid; gap: 24px; align-content: start; }\n"
            ".topbar { display: flex; align-items: center; justify-content: space-between; gap: 16px; }\n"
            ".eyebrow { margin: 0 0 8px; color: #2563eb; font-size: 13px; font-weight: 700; text-transform: uppercase; }\n"
            "h1 { margin: 0; font-size: 36px; line-height: 1.1; }\n"
            "button { border: 0; border-radius: 8px; background: #2563eb; color: white; padding: 10px 14px; font-weight: 700; }\n"
            ".prompt { max-width: 900px; font-size: 18px; line-height: 1.55; margin: 0; }\n"
            ".statusGrid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }\n"
            ".metric { background: white; border: 1px solid #dbe4ef; border-radius: 8px; padding: 16px; display: grid; gap: 8px; }\n"
            ".metric span { color: #607286; font-size: 13px; }\n"
            ".metric strong { font-size: 24px; }\n"
            ".metric.blue { border-top: 4px solid #2563eb; }\n"
            ".metric.green { border-top: 4px solid #0f766e; }\n"
            ".metric.amber { border-top: 4px solid #f59e0b; }\n"
            ".workbench { display: grid; grid-template-columns: minmax(0, 1.5fr) minmax(280px, 0.8fr); gap: 16px; }\n"
            ".workbench article { background: white; border: 1px solid #dbe4ef; border-radius: 8px; padding: 20px; }\n"
            "h2 { margin-top: 0; font-size: 18px; }\n"
            "li { margin: 8px 0; }\n"
            "@media (max-width: 760px) { .appShell { grid-template-columns: 1fr; } .sidebar { display: none; } .workspace { padding: 20px; } .topbar, .workbench, .statusGrid { grid-template-columns: 1fr; display: grid; } }\n"
        ),
    }

    written: list[str] = []
    for relative_path, content in files.items():
        target = project_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(relative_path)
    return written


def _write_vite_template(
    project_path: Path,
    *,
    app_name: str,
    prompt: str,
) -> list[str]:
    src_dir = project_path / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    safe_title = escape(app_name.strip() or "Code Bridge App")
    safe_prompt = escape(prompt.strip())
    package = {
        "name": _package_name(app_name),
        "version": "0.1.0",
        "private": True,
        "type": "module",
        "scripts": {
            "dev": "vite --host 0.0.0.0",
            "build": "tsc -b && vite build",
            "preview": "vite preview --host 0.0.0.0",
        },
        "dependencies": {
            "@vitejs/plugin-react": "latest",
            "vite": "latest",
            "typescript": "latest",
            "react": "latest",
            "react-dom": "latest",
        },
        "devDependencies": {
            "@types/react": "latest",
            "@types/react-dom": "latest",
        },
    }
    files = {
        "codebridge.agent.json": json.dumps(
            _agent_manifest(app_name, prompt, "vite", dev_port=5173),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        "package.json": json.dumps(package, indent=2, ensure_ascii=False) + "\n",
        "index.html": "<!doctype html>\n<html lang=\"en\">\n  <head>\n    <meta charset=\"UTF-8\" />\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />\n    <title>Code Bridge App</title>\n  </head>\n  <body>\n    <div id=\"root\"></div>\n    <script type=\"module\" src=\"/src/main.tsx\"></script>\n  </body>\n</html>\n",
        "tsconfig.json": json.dumps(
            {
                "compilerOptions": {
                    "target": "ES2020",
                    "useDefineForClassFields": True,
                    "lib": ["DOM", "DOM.Iterable", "ES2020"],
                    "allowJs": False,
                    "skipLibCheck": True,
                    "esModuleInterop": True,
                    "allowSyntheticDefaultImports": True,
                    "strict": True,
                    "forceConsistentCasingInFileNames": True,
                    "module": "ESNext",
                    "moduleResolution": "Node",
                    "resolveJsonModule": True,
                    "isolatedModules": True,
                    "noEmit": True,
                    "jsx": "react-jsx",
                },
                "include": ["src"],
            },
            indent=2,
        )
        + "\n",
        "vite.config.ts": "import { defineConfig } from 'vite';\nimport react from '@vitejs/plugin-react';\n\nexport default defineConfig({ plugins: [react()] });\n",
        ".gitignore": "node_modules\ndist\n.env*.local\n.DS_Store\n",
        "README.md": (
            f"# {app_name}\n\n"
            "Seeded by Code Bridge Agent Cockpit.\n\n"
            "## Initial prompt\n\n"
            f"{prompt}\n\n"
            "## Local commands\n\n"
            "- `npm install`\n"
            "- `npm run dev`\n"
            "- `npm run build`\n"
        ),
        "docs/agent-brief.md": _agent_brief(app_name, prompt),
        "src/main.tsx": (
            "import React from 'react';\n"
            "import { createRoot } from 'react-dom/client';\n"
            "import './styles.css';\n\n"
            "const workItems = [\n"
            "  { label: 'Product flow', value: 'Draft', tone: 'blue' },\n"
            "  { label: 'Data model', value: 'Open', tone: 'green' },\n"
            "  { label: 'Verification', value: 'Ready', tone: 'amber' },\n"
            "];\n\n"
            "function App() {\n"
            "  return (\n"
            "    <main className=\"appShell\">\n"
            "      <aside className=\"sidebar\">\n"
            f"        <div className=\"brand\">{safe_title}</div>\n"
            "        <nav><span className=\"active\">Workspace</span><span>Tasks</span><span>Preview</span></nav>\n"
            "      </aside>\n"
            "      <section className=\"workspace\">\n"
            "        <header className=\"topbar\">\n"
            "          <div><p className=\"eyebrow\">Initial agent brief</p>"
            f"<h1>{safe_title}</h1></div>\n"
            "          <button>Run build</button>\n"
            "        </header>\n"
            f"        <p className=\"prompt\">{safe_prompt}</p>\n"
            "        <section className=\"statusGrid\">\n"
            "          {workItems.map((item) => <article className={`metric ${item.tone}`} key={item.label}><span>{item.label}</span><strong>{item.value}</strong></article>)}\n"
            "        </section>\n"
            "      </section>\n"
            "    </main>\n"
            "  );\n"
            "}\n\n"
            "createRoot(document.getElementById('root')!).render(<React.StrictMode><App /></React.StrictMode>);\n"
        ),
        "src/styles.css": (
            "* { box-sizing: border-box; }\n"
            "body { margin: 0; font-family: Arial, Helvetica, sans-serif; background: #f4f7fb; color: #12212f; }\n"
            ".appShell { min-height: 100vh; display: grid; grid-template-columns: 240px 1fr; }\n"
            ".sidebar { background: #12212f; color: #eef6ff; padding: 24px; }\n"
            ".brand { font-weight: 800; font-size: 20px; margin-bottom: 32px; }\n"
            "nav { display: grid; gap: 8px; }\n"
            "nav span { color: #b8c7d9; padding: 10px 12px; border-radius: 8px; }\n"
            "nav .active { background: #1f3a52; color: #ffffff; }\n"
            ".workspace { padding: 32px; display: grid; gap: 24px; align-content: start; }\n"
            ".topbar { display: flex; align-items: center; justify-content: space-between; gap: 16px; }\n"
            ".eyebrow { margin: 0 0 8px; color: #2563eb; font-size: 13px; font-weight: 700; text-transform: uppercase; }\n"
            "h1 { margin: 0; font-size: 36px; line-height: 1.1; }\n"
            "button { border: 0; border-radius: 8px; background: #2563eb; color: white; padding: 10px 14px; font-weight: 700; }\n"
            ".prompt { max-width: 900px; font-size: 18px; line-height: 1.55; margin: 0; }\n"
            ".statusGrid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }\n"
            ".metric { background: white; border: 1px solid #dbe4ef; border-radius: 8px; padding: 16px; display: grid; gap: 8px; }\n"
            ".metric span { color: #607286; font-size: 13px; }\n"
            ".metric strong { font-size: 24px; }\n"
            ".metric.blue { border-top: 4px solid #2563eb; }\n"
            ".metric.green { border-top: 4px solid #0f766e; }\n"
            ".metric.amber { border-top: 4px solid #f59e0b; }\n"
            "@media (max-width: 760px) { .appShell { grid-template-columns: 1fr; } .sidebar { display: none; } .workspace { padding: 20px; } .topbar, .statusGrid { grid-template-columns: 1fr; display: grid; } }\n"
        ),
    }

    written: list[str] = []
    for relative_path, content in files.items():
        target = project_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(relative_path)
    return written


def _write_flutter_template(
    project_path: Path,
    *,
    app_name: str,
    prompt: str,
    package_name: str | None = None,
) -> list[str]:
    lib_dir = project_path / "lib"
    test_dir = project_path / "test"
    lib_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    package_name = package_name or _dart_package_name(app_name)
    safe_title = _dart_string_literal(app_name.strip() or "Code Bridge App")
    safe_prompt = _dart_string_literal(prompt.strip())
    files = {
        "codebridge.agent.json": json.dumps(
            _agent_manifest(app_name, prompt, "flutter", dev_port=0),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        "pubspec.yaml": (
            f"name: {package_name}\n"
            "description: Seeded by Code Bridge Agent Cockpit.\n"
            "publish_to: 'none'\n"
            "version: 0.1.0+1\n\n"
            "environment:\n"
            "  sdk: '>=3.4.0 <4.0.0'\n\n"
            "dependencies:\n"
            "  flutter:\n"
            "    sdk: flutter\n\n"
            "dev_dependencies:\n"
            "  flutter_test:\n"
            "    sdk: flutter\n"
            "  flutter_lints: ^4.0.0\n\n"
            "flutter:\n"
            "  uses-material-design: true\n"
        ),
        "analysis_options.yaml": "include: package:flutter_lints/flutter.yaml\n",
        ".gitignore": ".dart_tool\nbuild\n.flutter-plugins\n.flutter-plugins-dependencies\n.pub-cache\n.pub\n.DS_Store\n",
        "README.md": (
            f"# {app_name}\n\n"
            "Seeded by Code Bridge Agent Cockpit.\n\n"
            "## Initial prompt\n\n"
            f"{prompt}\n\n"
            "## Local commands\n\n"
            "- `flutter create --platforms=android,ios .`\n"
            "- `flutter pub get`\n"
            "- `flutter run`\n"
            "- `flutter test`\n"
        ),
        "docs/agent-brief.md": _agent_brief(app_name, prompt),
        "lib/main.dart": (
            "import 'package:flutter/material.dart';\n\n"
            "void main() {\n"
            "  runApp(const SeedApp());\n"
            "}\n\n"
            "class SeedApp extends StatelessWidget {\n"
            "  const SeedApp({super.key});\n\n"
            "  @override\n"
            "  Widget build(BuildContext context) {\n"
            "    return MaterialApp(\n"
            f"      title: {safe_title},\n"
            "      debugShowCheckedModeBanner: false,\n"
            "      theme: ThemeData(\n"
            "        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF2563EB)),\n"
            "        useMaterial3: true,\n"
            "      ),\n"
            "      home: const SeedHome(),\n"
            "    );\n"
            "  }\n"
            "}\n\n"
            "class SeedHome extends StatelessWidget {\n"
            "  const SeedHome({super.key});\n\n"
            "  @override\n"
            "  Widget build(BuildContext context) {\n"
            "    final colorScheme = Theme.of(context).colorScheme;\n"
            "    return Scaffold(\n"
            "      appBar: AppBar(title: const Text('Agent workspace')),\n"
            "      body: SafeArea(\n"
            "        child: ListView(\n"
            "          padding: const EdgeInsets.all(20),\n"
            "          children: [\n"
            f"            Text({safe_title}, style: Theme.of(context).textTheme.headlineMedium),\n"
            "            const SizedBox(height: 12),\n"
            f"            Text({safe_prompt}),\n"
            "            const SizedBox(height: 24),\n"
            "            Wrap(\n"
            "              spacing: 12,\n"
            "              runSpacing: 12,\n"
            "              children: const [\n"
            "                _Metric(label: 'Product flow', value: 'Draft'),\n"
            "                _Metric(label: 'Data model', value: 'Open'),\n"
            "                _Metric(label: 'Verification', value: 'Ready'),\n"
            "              ],\n"
            "            ),\n"
            "            const SizedBox(height: 24),\n"
            "            Card(\n"
            "              child: Padding(\n"
            "                padding: const EdgeInsets.all(16),\n"
            "                child: Column(\n"
            "                  crossAxisAlignment: CrossAxisAlignment.start,\n"
            "                  children: [\n"
            "                    Text('Primary workflow', style: Theme.of(context).textTheme.titleMedium),\n"
            "                    const SizedBox(height: 8),\n"
            "                    const Text('Replace this seed panel with the first production workflow.'),\n"
            "                  ],\n"
            "                ),\n"
            "              ),\n"
            "            ),\n"
            "            const SizedBox(height: 16),\n"
            "            FilledButton.icon(\n"
            "              onPressed: () {},\n"
            "              icon: const Icon(Icons.play_arrow),\n"
            "              label: const Text('Run build preflight'),\n"
            "              style: FilledButton.styleFrom(backgroundColor: colorScheme.primary),\n"
            "            ),\n"
            "          ],\n"
            "        ),\n"
            "      ),\n"
            "    );\n"
            "  }\n"
            "}\n\n"
            "class _Metric extends StatelessWidget {\n"
            "  final String label;\n"
            "  final String value;\n\n"
            "  const _Metric({required this.label, required this.value});\n\n"
            "  @override\n"
            "  Widget build(BuildContext context) {\n"
            "    return SizedBox(\n"
            "      width: 180,\n"
            "      child: Card(\n"
            "        child: Padding(\n"
            "          padding: const EdgeInsets.all(16),\n"
            "          child: Column(\n"
            "            crossAxisAlignment: CrossAxisAlignment.start,\n"
            "            children: [\n"
            "              Text(label),\n"
            "              const SizedBox(height: 8),\n"
            "              Text(value, style: Theme.of(context).textTheme.titleLarge),\n"
            "            ],\n"
            "          ),\n"
            "        ),\n"
            "      ),\n"
            "    );\n"
            "  }\n"
            "}\n"
        ),
        "test/widget_test.dart": (
            "import 'package:flutter_test/flutter_test.dart';\n"
            "import 'package:"
            f"{package_name}/main.dart';\n\n"
            "void main() {\n"
            "  testWidgets('renders seeded app', (tester) async {\n"
            "    await tester.pumpWidget(const SeedApp());\n"
            f"    expect(find.text({safe_title}), findsOneWidget);\n"
            "  });\n"
            "}\n"
        ),
    }

    written: list[str] = []
    for relative_path, content in files.items():
        target = project_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(relative_path)
    return written


def _materialize_flutter_platforms(
    project_path: Path,
    *,
    package_name: str,
) -> dict[str, Any]:
    flutter_path = shutil.which("flutter")
    command = [
        "flutter",
        "create",
        "--platforms=android,ios",
        "--project-name",
        package_name,
        "--no-pub",
        ".",
    ]
    if flutter_path is None:
        return {
            "status": "skipped",
            "attempted": False,
            "available": False,
            "reason": "flutter CLI not found",
            "command": " ".join(command),
            "platforms": [],
        }

    try:
        completed = subprocess.run(
            [flutter_path, *command[1:]],
            cwd=str(project_path),
            text=True,
            capture_output=True,
            timeout=90,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "failed",
            "attempted": True,
            "available": True,
            "reason": "flutter create timed out",
            "command": " ".join(command),
            "stdout_tail": _tail_text(exc.stdout),
            "stderr_tail": _tail_text(exc.stderr),
            "platforms": _existing_flutter_platforms(project_path),
        }
    except OSError as exc:
        return {
            "status": "failed",
            "attempted": True,
            "available": True,
            "reason": str(exc),
            "command": " ".join(command),
            "platforms": _existing_flutter_platforms(project_path),
        }

    platforms = _existing_flutter_platforms(project_path)
    return {
        "status": "created" if completed.returncode == 0 else "failed",
        "attempted": True,
        "available": True,
        "return_code": completed.returncode,
        "command": " ".join(command),
        "stdout_tail": _tail_text(completed.stdout),
        "stderr_tail": _tail_text(completed.stderr),
        "platforms": platforms,
    }


def _existing_flutter_platforms(project_path: Path) -> list[str]:
    return [
        name
        for name in ("android", "ios")
        if (project_path / name).is_dir()
    ]


def _tail_text(value: str | bytes | None, *, limit: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = value
    return text[-limit:]


def _tsx_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _dart_string_literal(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _initial_task_specs(app_name: str, prompt: str) -> list[tuple[str, str, int]]:
    return [
        (
            f"Shape {app_name} product flow",
            f"Turn the initial prompt into screens, states, and acceptance criteria: {prompt}",
            5,
        ),
        (
            "Implement the first usable workflow",
            "Replace the seed shell with production UI, state handling, and data wiring.",
            4,
        ),
        (
            "Verify build and preview",
            "Run install, lint, build, and preview checks, then attach the results to this run.",
            3,
        ),
    ]


def _mime_for_path(path: str) -> str:
    if path.endswith(".json"):
        return "application/json"
    if path.endswith(".md"):
        return "text/markdown"
    if path.endswith(".css"):
        return "text/css"
    if path.endswith(".tsx"):
        return "text/tsx"
    if path.endswith(".dart"):
        return "text/x-dart"
    if path.endswith(".yaml") or path.endswith(".yml"):
        return "application/yaml"
    return "text/plain"
