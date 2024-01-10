#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ================================================== #
# This file is a part of PYGPT package               #
# Website: https://pygpt.net                         #
# GitHub:  https://github.com/szczyglis-dev/py-gpt   #
# MIT License                                        #
# Created By  : Marcin Szczygliński                  #
# Updated Date: 2024.01.10 10:00:00                  #
# ================================================== #

from pygpt_net.utils import trans


class Dictionary:
    def __init__(self, window=None):
        """
        Dictionary field handler

        :param window: Window instance
        """
        self.window = window

    def apply(self, parent_id: str, key: str, option: dict):
        """
        Apply values to dictionary

        :param parent_id: Options parent ID
        :param key: Option key
        :param option: Option data
        """
        values = list(option["value"])
        self.window.ui.config[parent_id][key].items = values  # replace model data list
        self.window.ui.config[parent_id][key].model.updateData(values)  # update model data

    def apply_row(self, parent_id: str, key: str, values: dict, idx: int):
        """
        Apply data values to dictionary

        :param parent_id: Options parent ID
        :param key: Option key
        :param values: dictionary data values
        :param idx: row index
        """
        self.window.ui.config[parent_id][key].update_item(idx, values)

    def get_value(self, parent_id: str, key: str, option: dict) -> list:
        """
        Get dictionary values

        :param parent_id: Options parent ID
        :param key: Option key
        :param option: Option data
        :return: list with dictionary values
        """
        return self.window.ui.config[parent_id][key].model.items

    def delete_item(self, parent_object, id: str, force: bool = False):
        """
        Show delete item (from dict list) confirmation dialog or executes delete

        :param parent_object: parent object
        :param id: item id
        :param force: force delete
        """
        if not force:
            self.window.ui.dialogs.confirm('settings.dict.delete', id, trans('settings.dict.delete.confirm'),
                                           parent_object)
            return

        # delete item
        if parent_object is not None:
            parent_object.delete_item_execute(id)

    def to_options(self, parent_id: str, option: dict) -> dict:
        """
        Convert dictionary items option to options

        :param parent_id: Parent ID
        :param option: dictionary items option
        :return: options dict
        """
        if "keys" not in option:
            return {}
        options = {}
        for key in option["keys"]:
            item = option["keys"][key]
            if isinstance(item, str):
                item = {
                    'label': parent_id + '.' + key,
                    'type': 'text',
                }
            options[key] = item
            if options[key]["type"] == "text":
                options[key]["type"] = "textarea"  # convert text fields to textarea fields
            if 'label' not in options[key]:
                options[key]['label'] = key
                options[key]['label'] = parent_id + '.' + options[key]['label']
        return options

    def append_editor(self, id: str, option: dict, data: dict):
        """
        Apply dict item option sub-values to dict editor

        :param id: Owner object ID
        :param option: Option dict
        :param data: Option data
        """
        parent_id = "dictionary." + id
        sub_options = self.to_options(id, option)
        for key in sub_options:
            sub_option = sub_options[key]
            sub_option['value'] = data[key]
            self.window.controller.config.apply(parent_id, key, sub_option)

    def save_editor(self, option_key: str, parent: str, fields: dict):
        """
        Save dict editor (called from dict editor dialog save button)

        :param option_key: Option key
        :param parent: Parent ID
        :param fields: Fields dict
        """
        values = {}
        dict_id = parent + "." + option_key
        dialog_id = "dictionary." + parent + "." + option_key  # dictionary parent ID
        idx = self.window.ui.dialog['editor.' + dialog_id].idx  # editing record idx is stored in dialog idx
        for key in fields:
            value = self.window.controller.config.get_value("dictionary." + dict_id, key, fields[key])
            values[key] = value

        # update values in dictionary item on list in parent
        self.apply_row(parent, option_key, values, idx)

        # close dialog
        self.window.ui.dialog['editor.' + dialog_id].close()
