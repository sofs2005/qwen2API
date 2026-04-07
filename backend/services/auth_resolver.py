import asyncio
import logging
from backend.core.account_pool import AccountPool, Account
from backend.core.browser_engine import _new_browser
from backend.core.config import settings
from backend.core.account_pool import Account
import logging
import asyncio
import random
import string
import time
import json
import re
from typing import Optional
from camoufox.async_api import AsyncCamoufox

log = logging.getLogger(__name__)

BASE_URL = "https://chat.qwen.ai"

from backend.core.browser_engine import _new_browser

async def get_fresh_token(email: str, password: str) -> str:
    """如果提供了此功能，用 playwright 重新登录获取 Token，这里提供一个 mock 或抛错以防未实现"""
    raise NotImplementedError("Auto-login not fully implemented yet in the separated architecture")

def _gen_password(length=14):
    chars = string.ascii_letters + string.digits + "!@#$%^&*"
    while True:
        pwd = "".join(random.choices(chars, k=length))
        if (any(c.isupper() for c in pwd) and any(c.islower() for c in pwd)
                and any(c.isdigit() for c in pwd) and any(c in "!@#$%^&*" for c in pwd)):
            return pwd

def _gen_username():
    first = random.choice(["Alex", "Sam", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Jamie",
                            "Drew", "Avery", "Quinn", "Blake", "Sage", "Reese", "Dakota", "Emery"])
    last = random.choice(["Smith", "Brown", "Wilson", "Lee", "Chen", "Wang", "Kim", "Park",
                           "Davis", "Miller", "Garcia", "Martinez", "Anderson", "Taylor", "Thomas"])
    return f"{first} {last}"

MAIL_BASE = "https://mail.chatgpt.org.uk"

class _EmailSession:
    def __init__(self):
        from curl_cffi import requests as cffi_requests
        self._session = cffi_requests.Session(impersonate="chrome")
        self._session.headers.update({
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36 Edg/145.0.0.0",
            "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
        })
        self._current_token = ""
        self._token_expires_at = 0
        self._initialized = False

    def _init_session(self) -> bool:
        try:
            resp = self._session.get(f"{MAIL_BASE}/", timeout=15)
            if resp.status_code != 200:
                return False
            match = re.search(r'window\.__BROWSER_AUTH\s*=\s*(\{[^}]+\})', resp.text)
            if match:
                auth_data = json.loads(match.group(1))
                self._current_token = auth_data.get("token", "")
                self._token_expires_at = auth_data.get("expires_at", 0)
                self._initialized = True
                return True
            # 新增获取 Token 的 fallback 逻辑
            resp = self._session.get(f"{MAIL_BASE}/api/auth/token", timeout=15)
            if resp.status_code == 200:
                auth_data = resp.json()
                self._current_token = auth_data.get("token", "")
                self._token_expires_at = auth_data.get("expires_at", 0)
                self._initialized = True
                return True
            return False
        except Exception as e:
            log.warning(f"[MailSession] init error: {e}")
            return False

    def _ensure_token(self) -> bool:
        if not self._initialized or not self._current_token or time.time() > self._token_expires_at - 120:
            return self._init_session()
        return True

    def get_email(self) -> str:
        if not self._ensure_token():
            raise Exception("mail.chatgpt.org.uk: session init failed")
        resp = self._session.get(
            f"{MAIL_BASE}/api/generate-email",
            headers={"accept": "*/*", "referer": f"{MAIL_BASE}/",
                     "x-inbox-token": self._current_token},
            timeout=15,
        )
        if resp.status_code == 401 or resp.status_code == 403:
            self._initialized = False
            self._init_session()
            resp = self._session.get(
                f"{MAIL_BASE}/api/generate-email",
                headers={"accept": "*/*", "referer": f"{MAIL_BASE}/",
                         "x-inbox-token": self._current_token},
                timeout=15,
            )
        data = resp.json()
        if not data.get("success"):
            raise Exception(f"mail.chatgpt.org.uk: generate-email failed: {data}")
        email = str(data.get("data", {}).get("email", "")).strip()
        new_tok = data.get("auth", {}).get("token", "")
        if new_tok:
            self._current_token = new_tok
            self._token_expires_at = data.get("auth", {}).get("expires_at", 0)
        return email

    def poll_verify_link(self, email: str, timeout_sec: int = 300) -> str:
        keywords = ("qwen", "verify", "activate", "confirm", "aliyun", "alibaba", "qwenlm")
        log.info(f"[MailSession] Polling inbox for {email} (timeout {timeout_sec}s)...")
        deadline = time.time() + timeout_sec
        attempt = 0
        while time.time() < deadline:
            attempt += 1
            try:
                # 尝试获取最新 token，如果缺失的话
                if not self._current_token:
                    self._init_session()
                    
                resp = self._session.get(
                    f"{MAIL_BASE}/api/emails",
                    params={"email": email},
                    headers={"accept": "*/*", "referer": f"{MAIL_BASE}/",
                             "x-inbox-token": self._current_token},
                    timeout=15,
                )
                if resp.status_code == 401 or resp.status_code == 403:
                    self._initialized = False
                    self._init_session()
                    time.sleep(3)
                    continue
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("auth", {}).get("token"):
                        self._current_token = data["auth"]["token"]
                        self._token_expires_at = data["auth"].get("expires_at", 0)
                    emails_list = data.get("data", {}).get("emails", [])
                    log.info(f"[MailSession] 第{attempt}次轮询，收件箱邮件数: {len(emails_list)}")
                    for msg in emails_list:
                        subject = str(msg.get("subject", ""))
                        parts = []
                        for field in ("html_content", "content", "body", "html", "text", "raw"):
                            v = msg.get(field)
                            if v: parts.append(str(v))
                        for field in ("payload", "data", "message"):
                            v = msg.get(field)
                            if isinstance(v, dict): parts.extend(str(x) for x in v.values() if x)
                            elif isinstance(v, str) and v: parts.append(v)
                        combined = " ".join(parts)
                        combined = (combined.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                                    .replace("\\u003c", "<").replace("\\u003e", ">")
                                    .replace("\\u0026", "&").replace("\\/", "/"))
                        all_links = re.findall(r'https?://[^\s"\'<>\\,\)]+', combined)
                        for link in all_links:
                            link = link.rstrip(".,;)")
                            if any(kw in link.lower() for kw in keywords):
                                log.info(f"[MailSession] 找到验证链接: {link[:120]}...")
                                return link
                        if any(kw in subject.lower() for kw in keywords) and all_links:
                            return all_links[0]
                elif resp.status_code == 403:
                    data = resp.json()
                    if data.get("error") == "Mailbox access denied":
                        log.warning(f"[MailSession] 邮件API HTTP 403: Mailbox access denied. 可能是Token过期。重新初始化 Session...")
                        self._initialized = False
                        self._init_session()
                        time.sleep(3)
                        continue
                    else:
                        log.warning(f"[MailSession] 邮件API HTTP 403: {resp.text[:100]}")
                else:
                    log.warning(f"[MailSession] 邮件API HTTP {resp.status_code}: {resp.text[:100]}")
            except Exception as e:
                log.warning(f"[MailSession] 轮询异常: {e}")
            time.sleep(5)
        log.error("[MailSession] 超时：未收到验证邮件")
        return ""

class _AsyncMailClient:
    def __init__(self):
        self._sess: Optional[_EmailSession] = None
        self._email = ""

    async def __aenter__(self):
        self._sess = await asyncio.to_thread(_EmailSession)
        return self

    async def __aexit__(self, *args):
        pass

    async def generate_email(self) -> str:
        self._email = await asyncio.to_thread(self._sess.get_email)
        return self._email

    async def get_verify_link(self, timeout_sec: int = 300) -> str:
        return await asyncio.to_thread(self._sess.poll_verify_link, self._email, timeout_sec)

    async def get_verify_link_for_email(self, email: str, timeout_sec: int = 300) -> str:
        return await asyncio.to_thread(self._sess.poll_verify_link, email, timeout_sec)

async def register_qwen_account() -> Optional[Account]:
    log.info("[Register] ── 开始注册流程 ──")
    async with _AsyncMailClient() as mail_client:
        log.info("[Register] [1/7] 生成临时邮箱...")
        email = await mail_client.generate_email()
        password = _gen_password()
        username = _gen_username()
        log.info(f"[Register] [1/7] 邮箱: {email}  用户名: {username}")

        try:
            async with _new_browser() as browser:
                page = await browser.new_page()
                log.info(f"[Register] [2/7] 打开注册页面: {BASE_URL}/auth?action=signup")
                try:
                    await page.goto(f"{BASE_URL}/auth?action=signup", wait_until="domcontentloaded", timeout=60000)
                except Exception as e:
                    log.warning(f"[Register] [2/7] 页面加载异常: {e}")

                log.info("[Register] [3/7] 填写注册表单...")
                name_input = None
                for sel in ['input[placeholder*="Full Name"]', 'input[placeholder*="Name"]']:
                    try:
                        name_input = await page.wait_for_selector(sel, timeout=15000)
                        if name_input: break
                    except Exception:
                        pass
                if not name_input:
                    inputs = await page.query_selector_all('input')
                    name_input = inputs[0] if len(inputs) >= 4 else None
                if not name_input:
                    log.error("[Register] [3/7] 找不到姓名输入框，注册中止")
                    return None

                await name_input.click(); await name_input.fill(username)
                log.info(f"[Register] [3/7]  ✓ 姓名: {username}")
                email_input = await page.query_selector('input[placeholder*="Email"]')
                if not email_input:
                    inputs = await page.query_selector_all('input')
                    email_input = inputs[1] if len(inputs) >= 2 else None
                if email_input: await email_input.click(); await email_input.fill(email)
                log.info(f"[Register] [3/7]  ✓ 邮箱: {email}")

                pwd_input = await page.query_selector('input[placeholder*="Password"]:not([placeholder*="Again"])')
                if not pwd_input:
                    inputs = await page.query_selector_all('input')
                    pwd_input = inputs[2] if len(inputs) >= 3 else None
                if pwd_input: await pwd_input.click(); await pwd_input.fill(password)

                confirm_input = await page.query_selector('input[placeholder*="Again"]')
                if not confirm_input:
                    inputs = await page.query_selector_all('input')
                    confirm_input = inputs[3] if len(inputs) >= 4 else None
                if confirm_input: await confirm_input.click(); await confirm_input.fill(password)
                log.info("[Register] [3/7]  ✓ 密码已填写")

                checkbox = await page.query_selector('input[type="checkbox"]')
                if checkbox and not await checkbox.is_checked(): await checkbox.click()
                else:
                    agree = await page.query_selector('text=I agree')
                    if agree: await agree.click()
                log.info("[Register] [3/7]  ✓ 同意条款")

                log.info("[Register] [4/7] 提交注册表单...")
                await asyncio.sleep(1)
                submit = await page.query_selector('button:has-text("Create Account")') or await page.query_selector('button[type="submit"]')
                if submit: await submit.click()
                log.info("[Register] [4/7] 已点击提交，等待页面跳转（6s）...")
                await asyncio.sleep(6)

                url_after = page.url
                log.info(f"[Register] [4/7] 提交后URL: {url_after}")

                # Check if already logged in (redirected to main page)
                token = None
                if BASE_URL in url_after and "auth" not in url_after:
                    log.info("[Register] [5/7] 已跳转主页，尝试直接获取token...")
                    await asyncio.sleep(3)
                    token = await page.evaluate("localStorage.getItem('token')")
                    if token:
                        log.info("[Register] [5/7] ✓ 注册后直接获取到token，跳过邮件验证")

                # If no token yet, try explicit login with email+password (faster than email poll)
                if not token:
                    log.info("[Register] [5/7] 尝试用账号密码直接登录...")
                    try:
                        await page.goto(f"{BASE_URL}/auth", wait_until="domcontentloaded", timeout=30000)
                        await asyncio.sleep(3)
                        li_email = await page.query_selector('input[placeholder*="Email"]')
                        if li_email: await li_email.fill(email)
                        li_pwd = await page.query_selector('input[type="password"]')
                        if li_pwd: await li_pwd.fill(password)
                        li_btn = await page.query_selector('button:has-text("Log in")') or await page.query_selector('button[type="submit"]')
                        if li_btn: await li_btn.click()
                        await asyncio.sleep(8)
                        token = await page.evaluate("localStorage.getItem('token')")
                        if token:
                            log.info("[Register] [5/7] ✓ 直接登录成功，获取到token")
                    except Exception as e:
                        log.warning(f"[Register] [5/7] 直接登录失败: {e}")

                # If still no token, poll email for verification link
                if not token:
                    log.info("[Register] [6/7] 等待验证邮件（最多5分钟）...")
                    verify_link = await mail_client.get_verify_link(timeout_sec=300)

                    if not verify_link:
                        log.error("[Register] [6/7] 未收到验证邮件，注册失败")
                        return None

                    log.info(f"[Register] [6/7] ✓ 收到验证链接，访问中...")
                    try:
                        await page.goto(verify_link, wait_until="domcontentloaded", timeout=30000)
                    except Exception: pass
                    await asyncio.sleep(6)
                    token = await page.evaluate("localStorage.getItem('token')")
                    log.info(f"[Register] [6/7] 验证后URL: {page.url}")

                    # Login after verification
                    if not token:
                        log.info("[Register] [6/7] 验证链接后尝试登录...")
                        try:
                            await page.goto(f"{BASE_URL}/auth", wait_until="domcontentloaded", timeout=30000)
                            await asyncio.sleep(3)
                            li_email = await page.query_selector('input[placeholder*="Email"]')
                            if li_email: await li_email.fill(email)
                            li_pwd = await page.query_selector('input[type="password"]')
                            if li_pwd: await li_pwd.fill(password)
                            li_btn = await page.query_selector('button:has-text("Log in")') or await page.query_selector('button[type="submit"]')
                            if li_btn: await li_btn.click()
                            await asyncio.sleep(8)
                            token = await page.evaluate("localStorage.getItem('token')")
                            if token:
                                log.info("[Register] [6/7] ✓ 验证后登录成功")
                        except Exception: pass

                if not token:
                    log.error("[Register] 所有方法均无法获取token，注册失败")
                    return None

                log.info("[Register] [7/7] 提取 cookies...")
                all_cookies = await page.context.cookies()
                cookie_str = "; ".join(f"{c.get('name','')}={c.get('value','')}" for c in all_cookies if "qwen" in c.get("domain", ""))
                log.info(f"[Register] ✓ 注册完成: {email}")
                return Account(email=email, password=password, token=token, cookies=cookie_str, username=username, activation_pending=False)
        except Exception as e:
            import traceback
            log.error(f"[Register] 注册异常: {e}\n{traceback.format_exc()}")
            return None

async def activate_account(acc: Account) -> bool:
    """Use API to poll for Qwen activation link, click it, then login to get a fresh token. Returns True on success."""
    log.info(f"[Activate] 开始激活 {acc.email}，通过 API 轮询邮箱...")
    try:
        async with _AsyncMailClient() as mail_client:
            verify_link = await mail_client.get_verify_link_for_email(acc.email, timeout_sec=120)
            
        if not verify_link:
            log.warning(f"[Activate] {acc.email} 邮箱未找到激活链接")
            return False

        log.info(f"[Activate] 找到激活链接: {verify_link[:120]}")

        async with _new_browser() as browser:
            page = await browser.new_page()
            
            # Visit the activation link
            try:
                await page.goto(verify_link, wait_until="networkidle", timeout=30000)
            except Exception:
                try:
                    await page.goto(verify_link, wait_until="domcontentloaded", timeout=15000)
                except Exception:
                    pass
            await asyncio.sleep(5)
            token = await page.evaluate("localStorage.getItem('token')")
            log.info(f"[Activate] 访问激活链接后 URL={page.url}, token={'有' if token else '无'}")

            # If no token yet, try logging in
            if not token and acc.password:
                try:
                    await page.goto(f"{BASE_URL}/auth", wait_until="domcontentloaded", timeout=30000)
                    await asyncio.sleep(3)
                    li_email = await page.query_selector('input[placeholder*="Email"]')
                    if li_email:
                        await li_email.fill(acc.email)
                    li_pwd = await page.query_selector('input[type="password"]')
                    if li_pwd:
                        await li_pwd.fill(acc.password)
                    li_btn = (await page.query_selector('button:has-text("Log in")') or
                              await page.query_selector('button[type="submit"]'))
                    if li_btn:
                        await li_btn.click()
                    await asyncio.sleep(8)
                    token = await page.evaluate("localStorage.getItem('token')")
                except Exception as e:
                    log.warning(f"[Activate] 激活后登录异常: {e}")

            if token:
                acc.token = token
                acc.valid = True
                acc.activation_pending = False
                log.info(f"[Activate] {acc.email} 激活成功，token已更新")
                return True
            log.warning(f"[Activate] {acc.email} 激活后未能获取token")
            return False
    except Exception as e:
        log.error(f"[Activate] {acc.email} 激活异常: {e}")
        return False

class AuthResolver:
    """自动登录并提取 Token，在检测到 401 时自动自愈凭证"""
    def __init__(self, pool: AccountPool):
        self.pool = pool

    async def auto_heal_account(self, acc: Account):
        """Background task to refresh token. If successful, marks account valid.
        If refresh fails or account is pending activation, tries to activate via email."""
        try:
            ok = await self.refresh_token(acc)
            if ok:
                if not getattr(acc, 'activation_pending', False):
                    acc.valid = True
                    await self.pool.save()
                    log.info(f"[BGRefresh] {acc.email} token刷新成功，账号已恢复")
                    return
                # Token refreshed but account still needs activation — don't mark valid yet
                log.info(f"[BGRefresh] {acc.email} token刷新成功，但账号仍待激活，继续尝试激活...")
            else:
                log.warning(f"[BGRefresh] {acc.email} 刷新失败，尝试邮件激活...")
            
            activated = await activate_account(acc)
            if activated:
                acc.activation_pending = False
                acc.valid = True
                await self.pool.save()  # 持久化激活状态，重启后不再重复激活
                log.info(f"[BGRefresh] {acc.email} 邮件激活成功，账号已恢复")
            else:
                log.warning(f"[BGRefresh] {acc.email} 激活失败，账号保持失效")
        except Exception as e:
            log.warning(f"[BGRefresh] {acc.email} 自动自愈异常: {e}")

    async def refresh_token(self, acc: Account) -> bool:
        """Re-login with email+password to get a fresh token. Returns True on success."""
        if not acc.email or not acc.password:
            log.warning(f"[Refresh] 账号 {acc.email} 无密码，无法刷新")
            return False
            
        log.info(f"[Refresh] 正在为 {acc.email} 刷新 token...")
        try:
            async with _new_browser() as browser:
                page = await browser.new_page()
                try:
                    await page.goto(f"{BASE_URL}/auth", wait_until="domcontentloaded", timeout=30000)
                except Exception:
                    pass
                await asyncio.sleep(3)
                
                li_email = await page.query_selector('input[placeholder*="Email"]')
                if li_email:
                    await li_email.fill(acc.email)
                li_pwd = await page.query_selector('input[type="password"]')
                if li_pwd:
                    await li_pwd.fill(acc.password)
                li_btn = (await page.query_selector('button:has-text("Log in")') or
                          await page.query_selector('button[type="submit"]'))
                if li_btn:
                    await li_btn.click()
                await asyncio.sleep(8)
                new_token = await page.evaluate("localStorage.getItem('token')")
                if new_token and new_token != acc.token:
                    old_prefix = acc.token[:20] if acc.token else "空"
                    acc.token = new_token
                    acc.valid = True
                    await self.pool.save()
                    log.info(f"[Refresh] {acc.email} token 已更新 ({old_prefix}... → {new_token[:20]}...)")
                    return True
                elif new_token == acc.token:
                    # Token same but might still be valid — mark valid again
                    acc.valid = True
                    log.info(f"[Refresh] {acc.email} token 未变化，重新标记有效")
                    return True
                else:
                    log.warning(f"[Refresh] {acc.email} 登录后未获取到token，URL={page.url}")
                    return False
        except Exception as e:
            log.error(f"[Refresh] {acc.email} 刷新异常: {e}")
            return False
