from win32 import win32service
from win32.lib import win32serviceutil
from multiprocessing import Process
import os
import sys
import logging
import logging.config

class Win32Svc (win32serviceutil.ServiceFramework):
    _svc_name_ = "LabelStudioFrontend"
    _svc_display_name_ = "LabelStudio Frontend Service"

    def __init__(self, *args):
        super().__init__(*args)

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        self.process.terminate()
        self.ReportServiceStatus(win32service.SERVICE_STOPPED)

    def SvcDoRun(self):
        self.process = Process(target=self.main)
        self.process.start()
        self.process.join()

    def main(self):
        sys.argv = ['service.py', 'runserver', '--noreload']
        d = os.path.dirname(__file__)
        workspace = os.path.join(d, '..\\..\\..\\workspace')
        os.environ.setdefault('LABEL_STUDIO_WORKSPACE', workspace)
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings.label_studio')
        # os.environ.setdefault('DEBUG', 'True')

        logging.config.dictConfig({
            "version": 1,
            "formatters": {
                "standard": {
                    "format": "[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "DEBUG",
                    "stream": "ext://sys.stdout",
                    "formatter": "standard"
                },
                 "file": {
                    "level": "DEBUG",
                    "formatter": "standard",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": os.path.join(workspace, "frontend.log"),
                    "maxBytes": 10*1024*1024,
                    "backupCount": 5
                }
            },
           
            "root": {
                "level": "INFO",
                "handlers": [
                    "file"
                ],
                "propagate": True
            }
        })

        try:
            from django.core.management import execute_from_command_line
            from django.core.management.commands.runserver import Command as runserver
            from django.conf import settings
            runserver.default_port = settings.INTERNAL_PORT

        except ImportError as exc:
            raise ImportError(
                "Couldn't import Django. Are you sure it's installed and "
                "available on your PYTHONPATH environment variable? Did you "
                "forget to activate a virtual environment?"
            ) from exc
        execute_from_command_line(sys.argv)

if __name__ == '__main__':
    win32serviceutil.HandleCommandLine(Win32Svc)
    
