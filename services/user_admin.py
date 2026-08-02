from __future__ import annotations

import argparse
import getpass

from config.settings import load_settings
from services.app_store import AppStore
from services.security import AuthService


def main() -> None:
    """Run local account recovery operations without exposing them as web APIs."""

    parser = argparse.ArgumentParser(description="PaperDesk local account administration")
    subcommands = parser.add_subparsers(dest="command", required=True)
    reset = subcommands.add_parser("reset-password", help="reset one local password")
    reset.add_argument("username")
    subcommands.add_parser("list-users", help="list local usernames")
    args = parser.parse_args()

    settings = load_settings()
    store = AppStore(settings)
    auth = AuthService(settings, store)
    if args.command == "list-users":
        with store.connect() as connection:
            rows = connection.execute(
                "SELECT username, display_name, created_at FROM users ORDER BY username"
            ).fetchall()
        for row in rows:
            print(f"{row['username']}\t{row['display_name']}")
        return

    user = store.get_user_by_username(args.username)
    if user is None:
        raise SystemExit("unknown username")
    password = getpass.getpass("New password: ")
    confirmation = getpass.getpass("Confirm password: ")
    if password != confirmation:
        raise SystemExit("passwords do not match")
    auth.validate_password(password)
    store.update_password(user.user_id, auth.hasher.hash(password))
    store.delete_all_auth_sessions(user.user_id)
    print(f"Password reset for {user.username}; all sessions were revoked.")


if __name__ == "__main__":
    main()
