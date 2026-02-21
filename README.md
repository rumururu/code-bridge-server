# Code Bridge Server

AI 코딩 도우미를 위한 PC 서버입니다. 모바일 앱과 QR 코드로 페어링하여 사용합니다.

## 설치

### macOS / Linux
```bash
curl -fsSL https://raw.githubusercontent.com/rumururu/code-bridge-server/main/install/install.sh | bash
```

### Windows (PowerShell)
```powershell
iwr -useb https://raw.githubusercontent.com/rumururu/code-bridge-server/main/install/install.ps1 | iex
```

## 수동 설치

```bash
# 1. 클론
git clone https://github.com/rumururu/code-bridge-server.git
cd code-bridge-server

# 2. 가상환경 생성
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 서버 실행 (QR 코드 표시)
python main.py --show-qr
```

## 사용법

```bash
# QR 코드와 함께 서버 시작
python main.py --show-qr

# QR 코드만 표시 (서버 실행 안 함)
python main.py --qr-only

# 특정 호스트/포트 지정
python main.py --host 0.0.0.0 --port 8080 --show-qr
```

서버가 시작되면 터미널에 QR 코드가 표시됩니다. Code Bridge 모바일 앱으로 스캔하여 연결하세요.

## 기능

- **QR 코드 페어링**: 모바일 앱과 간편하게 연결
- **로컬 네트워크**: 같은 WiFi에서 직접 연결
- **Cloudflare 터널**: 외부 네트워크에서도 접속 가능 (선택)
- **Firebase 인증**: SSO 로그인 지원 (선택)

## API 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/api/health` | GET | 서버 상태 확인 |
| `/api/pair/qr` | GET | QR 코드 데이터 |
| `/api/pair/verify` | POST | 페어링 검증 |
| `/api/pair/status` | GET | 페어링 상태 |
| `/api/projects` | GET | 프로젝트 목록 |

## 요구사항

- Python 3.10+
- Git
- (선택) cloudflared - 외부 네트워크 접속용

## 라이선스

MIT License
