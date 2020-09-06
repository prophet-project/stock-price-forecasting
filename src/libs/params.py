import yaml

fd = open("params.yaml", 'r')

params = yaml.safe_load(fd)

fd.close()