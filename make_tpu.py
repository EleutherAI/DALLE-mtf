# TODO:
# make pip installable
# should be able to set a list of default names, and if you don't pass in a name,
# it will pick a name at random from the list


import os
import logging
import random
from tpunicorn.tpu import get_tpu


class TPUMaker:

    def __init__(self, debug_mode=True):
        # set defaults
        self.namelist = []
        self.tf_version = "1.15.2"
        self.preemptible_v8s = False
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG) if debug_mode else self.logger.setLevel(logging.INFO)
        self.project = None
        self.zone = None

    def make_tpu(self, size, name=None, tf_version=None, accelerator_type="v3", preemptible=True, zone=None,
                 project=None):

        project = self.project if project is None else project
        assert project is not None, "Please set default project with maketpu setproject projectname, or pass in a " \
                                    "project to maketpu.\n e.g, maketpu test 8 --project <projectname>"

        zone = self.zone if zone is None else zone
        assert zone is not None, "Please set default zone with maketpu setzone zonename, or pass in a " \
                                 "zone to maketpu.\n e.g, maketpu test 8 --zone <zonename>"

        # if making a v-8, set preemptible to false if this is the project's default
        if not self.preemptible_v8s and size == 8:
            preemptible = False
            self.logger.debug(
                "Setting preemptible to false, as this project's does not have access to preemptible v-8s")
        if preemptible:
            p = "--preemptible"
        else:
            p = ""

        # if no name is specified, pick one at random from the namelist
        if name is None:
            name = self.get_name()

        tf_version = self.tf_version if tf_version is None else tf_version

        command = f"gcloud compute tpus create {name} --zone {zone} --project {project} --network default --version {tf_version} --accelerator-type {accelerator_type}-{size} {p}"
        self.logger.info(command)
        os.system(command)

    def add_to_namelist(self, name):
        self.namelist.append(name)

    def set_project(self, project_name):
        self.project = project_name

    def set_zone(self, zone):
        self.zone = zone

    def tpu_exists(self, name):
        if get_tpu(name, project=self.project, silent=True) is None:
            return False
        else:
            return True

    def get_name(self):
        self.logger.debug("getting name")
        if self.namelist:
            available_names = self.namelist
            name = random.choice(available_names)
            x = 0
            while True:
                x += 1
                self.logger.debug(available_names)
                if not available_names:
                    raise Exception("All tpu names in default namelist already exist - please pass a name to "
                                    "maketpu or update default namelist")
                if self.tpu_exists(name):
                    self.logger.debug(f'TPU {name} exists')
                    available_names.remove(name)
                    name = random.choice(available_names)
                    self.logger.debug(f'trying {name}')
                else:
                    break
            self.logger.debug(f"got name {name}")
            return name
        else:
            raise Exception("No name specified and default namelist is empty")


if __name__ == "__main__":
    t = TPUMaker()
    t.set_project("youdreamof-1543654322305")
    t.set_zone("europe-west4-a")
    t.namelist = ["test"]
    t.tf_version = "2.4.0"
    t.make_tpu(32)
