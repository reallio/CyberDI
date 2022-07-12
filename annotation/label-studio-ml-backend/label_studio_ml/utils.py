import importlib
import importlib.util
import inspect
import os
import sys
import urllib
import hashlib
import requests
import logging
import io

from urllib.parse import urlparse
from PIL import Image
from colorama import Fore

if __name__ == 'utils':
    from model import LabelStudioMLBase
else:
    from .model import LabelStudioMLBase
from label_studio.core.utils.io import get_cache_dir, get_data_dir
from label_studio.core.utils.params import get_env


logger = logging.getLogger(__name__)


def get_all_classes_inherited_LabelStudioMLBase(script_file):
    names = set()
    abs_path = os.path.abspath(script_file)
    module_name = os.path.splitext(os.path.basename(script_file))[0]
    sys.path.append(os.path.dirname(abs_path))
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        print(Fore.RED + 'Can\'t import module "' + module_name + f'", reason: {e}.\n'
              'If you are looking for examples, you can find a dummy model.py here:\n' +
              Fore.LIGHTYELLOW_EX + 'https://labelstud.io/tutorials/dummy_model.html')
        module = None
        exit(-1)

    for name, obj in inspect.getmembers(module, inspect.isclass):
        if name == LabelStudioMLBase.__name__:
            continue
        if issubclass(obj, LabelStudioMLBase):
            names.add(name)
        for base in obj.__bases__:
            if LabelStudioMLBase.__name__ == base.__name__:
                names.add(name)
    sys.path.pop()
    names = list(names)
    return names


def get_single_tag_keys(parsed_label_config, control_type, object_type):
    """
    Gets parsed label config, and returns data keys related to the single control tag and the single object tag schema
    (e.g. one "Choices" with one "Text")
    :param parsed_label_config: parsed label config returned by "label_studio.misc.parse_config" function
    :param control_type: control tag str as it written in label config (e.g. 'Choices')
    :param object_type: object tag str as it written in label config (e.g. 'Text')
    :return: 3 string keys and 1 array of string labels: (from_name, to_name, value, labels)
    """
    assert len(parsed_label_config) == 1
    from_name, info = list(parsed_label_config.items())[0]
    assert info['type'] == control_type, 'Label config has control tag "<' + info['type'] + '>" but "<' + control_type + '>" is expected for this model.'  # noqa

    assert len(info['to_name']) == 1
    assert len(info['inputs']) == 1
    assert info['inputs'][0]['type'] == object_type
    to_name = info['to_name'][0]
    value = info['inputs'][0]['value']
    return from_name, to_name, value, info['labels']


def is_skipped(completion):
    if len(completion['annotations']) != 1:
        return False
    completion = completion['annotations'][0]
    return completion.get('skipped', False) or completion.get('was_cancelled', False)


def get_choice(completion):
    return completion['annotations'][0]['result'][0]['value']['choices'][0]


def get_image_local_path(url, image_cache_dir=None, project_dir=None, image_dir=None):
    return get_local_path(url, image_cache_dir, project_dir, get_env('HOSTNAME'), image_dir)


def get_local_path(url, cache_dir=None, project_dir=None, hostname=None, image_dir=None, access_token=None):
    is_local_file = url.startswith('/data/') and '?d=' in url
    is_uploaded_file = url.startswith('/data/upload')
    if image_dir is None:
        upload_dir = os.path.join(get_data_dir(), 'media', 'upload')
        image_dir = project_dir and os.path.join(project_dir, 'upload') or upload_dir

    # File reference created with --allow-serving-local-files option
    if is_local_file:
        filename, dir_path = url.split('/data/')[1].split('?d=')
        dir_path = str(urllib.parse.unquote(dir_path))
        filepath = os.path.join(dir_path, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)
        return filepath

    # File uploaded via import UI
    elif is_uploaded_file and os.path.exists(image_dir):
        project_id = url.split("/")[-2] # To retrieve project_id
        image_dir = os.path.join(image_dir, project_id)
        filepath = os.path.join(image_dir, os.path.basename(url))
        return filepath

    elif is_uploaded_file and hostname:
        url = hostname + url
        logger.info('Resolving url using hostname [' + hostname + '] from LSB: ' + url)

    elif is_uploaded_file:
        raise FileNotFoundError("Can't resolve url, neither hostname or project_dir passed: " + url)

    if is_uploaded_file and not access_token:
        raise FileNotFoundError("Can't access file, no access_token provided for Label Studio Backend")

    # File specified by remote URL - download and cache it
    cache_dir = cache_dir or get_cache_dir()
    parsed_url = urlparse(url)
    url_filename = os.path.basename(parsed_url.path)
    url_hash = hashlib.md5(url.encode()).hexdigest()[:6]
    filepath = os.path.join(cache_dir, url_hash + '__' + url_filename)
    if not os.path.exists(filepath):
        logger.info('Download {url} to {filepath}'.format(url=url, filepath=filepath))
        headers = {'Authorization': 'Token ' + access_token} if access_token else {}
        r = requests.get(url, stream=True, headers=headers)
        r.raise_for_status()
        with io.open(filepath, mode='wb') as fout:
            fout.write(r.content)
    return filepath


def get_image_size(filepath):
    return Image.open(filepath).size
