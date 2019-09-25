#!/usr/bin/env python

import json

with open("config.json", 'r') as load_f:
    load_dict = json.load(load_f)
    print(load_dict)

load_dict['server_port'] = load_dict['server_port'] + 1
print(load_dict)

with open("config.json", "w") as dump_f:
    json.dump(load_dict, dump_f)
