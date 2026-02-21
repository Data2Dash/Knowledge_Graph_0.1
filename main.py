# main.py
from __future__ import annotations

import sys
import traceback

from app.core.settings import get_settings
from app.core.logging import configure_logging


def main() -> None:
    """
    Application entrypoint (dev-friendly).

    - Loads Settings (supports ENV_FILE)
    - Ensures runtime directories exist
    - Configures logging once
    - Runs Streamlit UI (when executed as plain python)
    """
    try:
        settings = get_settings()
        settings.ensure_runtime_dirs()

        configure_logging(
            log_level=settings.LOG_LEVEL,
            log_dir=settings.resolved_log_dir(),
            log_json=settings.LOG_JSON,
            app_name=settings.APP_NAME,
        )

        # Import UI only after settings/logging are ready (avoids side effects)
        from app.ui.streamlit_app import run_app

        run_app()

    except Exception as e:
        print("\n=== App Startup Failure ===\n")
        print(str(e))
        print("\n--- Traceback ---\n")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()