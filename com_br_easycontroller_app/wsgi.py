import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE',
                      'com_br_easycontroller_app.settings')

app = get_wsgi_application()
