"""
Generated python code for component
Component flow is as follows
A. Local Execution
1. local_execution.py creates local proxy object, and calls its start method
2. local proxy calls component on_start method - to be implemented by developer
3. Developer may make calls to report_status, report_kpi_metrics and report_timeseries_metrics - these
are mapped to local proxy methods which print information on the console
4. Developer may make calls to completed method - this calls the on_complete method of the component
- Developer can implement this method to perform any cleanup tasks

B. Remote Execution
1a. xpresso Controller calls xprbuild/system/linux/run.sh, which calls the remote proxy class main function
1b. This creates remote proxy object, and calls its start method
2. Remote proxy calls component on_start method - to be implemented by developer
3. Developer may make calls to report_status, report_kpi_metrics and report_timeseries_metrics - these
are mapped to remote proxy methods which send the information to the xpresso controller
4. Developer may make calls to completed method - this sends completion information to the xpresso controller
after calling the on_complete method of the component
- Developer can implement this method to perform any cleanup tasks
"""

import os
from types import new_class
import psutil
import time
import sys
import json
import numpy as np
import pandas as pd
import logging.config
from datetime import datetime

from app.helper import *
from app.train_model import Model
from app.mail import send_email
from app.run_metadata import get_run_metadata
from app.data_summary import prep_data_understanding
from app.trigger_experiment import trigger_pipeline_run
from app.metrics import *

from xpresso.ai.core.data.versioning.controller_factory import VersionControllerFactory
from xpresso.ai.client.controller_client import ControllerClient

constants_config = json.load(open("/project/config/constants.json"))
ENABLE_LOCAL_EXECUTION = "enable_local_execution"

__author__ = "Rohit Kr Singh"

import ray
#num_cpus = psutil.cpu_count(logical=True)
num_cpus = constants_config['misc_params']['num_cpus_ray']
ray.init(num_cpus=num_cpus, ignore_reinit_error=True)
# ray.init(local_mode = True)
assert ray.is_initialized() == True
os.environ['RAY_IGNORE_UNHANDLED_ERRORS']='1'

if not os.path.exists("/output"):
    os.makedirs("/output")


class RfTraining():
    def __init__(self, xpresso_run_name, run_parameters, pipeline, use_case, ontology):
        """
        Constructor
            Note: Constructor cannot take any parameters
            Initialize all the required constants and data here
        """
        if self.str2bool(os.environ.get(ENABLE_LOCAL_EXECUTION, "False")):
            self.local_execution = True
        else:
            self.local_execution = False

        self.xpresso_run_name = xpresso_run_name
        self.CONF = run_parameters
        self.pipeline = pipeline
        self.use_case = use_case
        self.ontology = ontology
        self.start_time = time.time()


    def on_start(self):
        """
        This is the start method, which implements the actual component logic.
        Developers must implement their component logic here
        and call the completed method when done

        Args:
        """
        try:
            if (self.pipeline=='inference'):
                self.completed(success=True)

            # Add comment here
            # data_path = self.fetch_versioned_data()
            data_path = f"/pipeline/{self.use_case}_output/merge_sk_data/"
            filepath = data_path + "/rf_input_" + self.use_case + ".csv"

            data = pd.read_csv(filepath)

            drop_cols = [col for col in list(data.columns) if(('-999' in col) or ('unknown' in col) or ('Unknown' in col))]
            # Drop numerical columns having all values as -999
            drop_cols = list(set(drop_cols + [col for col in data.columns if (data[col]==-999).all()]))
        
            data.drop(drop_cols, axis=1, inplace=True)

            self.CONF['cols_used_in_modeling'] = list(data.columns)

            rf_params = self.CONF['model_params']
            # Data Understanding
            self.logger.info("Preparing & versioning Data Understanding for modeling data")
            prep_data_understanding(data, self.use_case, self.logger)

            y = data.pop('condition_flag')
            
            if self.use_case=='sales':
                cols_to_drop = ['First Name','Middle Name Initial','Last Name','Zip_Code']
                sales_df = data[cols_to_drop]
                X = data.drop(cols_to_drop, axis=1)
            else:
                data['CONSISTENT_MEMBER_ID'] = data['CONSISTENT_MEMBER_ID'].astype(int)
                cmids = data['CONSISTENT_MEMBER_ID'].to_list()
                X = data.set_index("CONSISTENT_MEMBER_ID", drop=True)
            
            # Model Training
            model = Model(X, y, rf_params, self.use_case, save_model=1, logger=self.logger)
            classifier = model.train()

            X_test, y_test = pd.DataFrame(), []
            if self.use_case == 'predictive' and self.CONF["train_final_model"] == 0:
                filepath = data_path + "/rf_input_" + self.use_case + "_test" + ".csv"
                test_data = pd.read_csv(filepath)
                test_data.drop(drop_cols, axis=1, inplace=True)
                #create X_test and y_test
                y_test = test_data.pop('condition_flag')
                cmids_test = test_data['CONSISTENT_MEMBER_ID'].to_list()
                X_test = test_data.drop(['CONSISTENT_MEMBER_ID'], axis=1) 
            
            #evaluate metrics
            report_dict, pred_df, report_dict_test, pred_df_test = evaluate_classification_metric(classifier, X, y, X_test, y_test, self.CONF, self.logger)
            
            #merging test report dict with train report dict
            report_dict = {**report_dict, **report_dict_test}
            kpi_dict = {}
            kpi_dict_keys = [key for key, val in report_dict.items() if kpi_conditions(val)]
            for k in kpi_dict_keys:
                kpi_dict[k] = report_dict[k][0]

            if self.use_case=='sales':
                pred_df = pd.concat([pred_df, sales_df], axis=1)
                del sales_df
            else:    
                pred_df.insert(loc=0, column='CONSISTENT_MEMBER_ID', value=cmids)
                if self.use_case == 'predictive' and not X_test.empty:
                    pred_df_test.insert(loc=0, column='CONSISTENT_MEMBER_ID', value=cmids_test)

            # Saving evaluation metric
            folderpath = f'/pipeline/{self.use_case}_output/'
            evaluation_metrics = pd.DataFrame(report_dict).T.reset_index()
            evaluation_metrics.columns = ['metrics', 'value']
            evaluation_metrics.to_csv(folderpath + "model_artifacts/evaluation_metrics.csv", index=False)
            evaluation_metrics.to_csv("/output/evaluation_metrics.csv", index=False)
            
            #saving predictions and probabilities
            pred_df.to_csv(folderpath + "model_artifacts/pred_prob.csv", index=False)
            pred_df.to_csv("/output/pred_prob.csv", index=False)
            if self.use_case == 'predictive' and self.CONF["train_final_model"] == 0:
                pred_df_test.to_csv(folderpath + "model_artifacts/pred_prob_test.csv", index=False)
                pred_df_test.to_csv("/output/pred_prob_test.csv", index=False)
            self.report_kpi_metrics(kpi_dict)

            # Explained Dashboard
            self.logger.info("Generating Explainer dashboard")
            flag = generate_dashboard(classifier, X, y, self.use_case, self.ontology, self.CONF, constants_config, self.logger)

            json.dump(self.CONF, open(f'/output/config_{self.use_case}.json','w'))
            # change push_exp to True if you have saved your model to OUTPUT_DIR, False if not
            self.completed(push_exp=True, success=True)
        except Exception as my_error:
            import traceback
            traceback.print_exc()
            _, _, exc_tb = sys.exc_info()
            self.logger.error(f'Error {my_error}, Line No {exc_tb.tb_lineno}')
            self.completed(success=False)


    def fetch_versioned_data(self):
        """ Fetch data from Data VCS from specified branch """
        try:
            branch_name = constants_config['branch_mappings'][f'branch_dict_{self.use_case}'][f'{self.use_case}_data_prep_b2']

            cc = ControllerClient()
            cc.login(constants_config["xpr_creds"]["username"],constants_config["xpr_creds"]["password"])

            version_controller_factory = VersionControllerFactory()
            version_controller = version_controller_factory.get_version_controller()
            data_path = version_controller.pull_dataset(repo_name="healthcare_models",
                                                        branch_name=branch_name,
                                                        output_type="files",
                                                        type="data")
            cc.logout()
            return data_path
        except Exception as my_error:
            _, _, exc_tb = sys.exc_info()
            self.logger.error(f'Error {my_error}, Line No {exc_tb.tb_lineno}')
            self.completed(success=False)


    def on_complete(self, push_exp=False, success=True):
        """
        This method is called by the local/remote proxy completed method
        Developers must implement any cleanup required in this method

        Args:
            push_exp: Whether to push the data present in the output folder
               to the versioning system. This is required once training is
               completed and model needs to be versioned
            success: Use to handle failure cases
        """
        # === Your completion code goes here ===
        elapsed_time_mins = (time.time() - self.start_time) / 60
        self.logger.info(f"Run for RF train completed in {round(elapsed_time_mins,4)} minutes with memory usage {psutil.virtual_memory()}")

        # Add comment here
        if (success==True):
            if not self.local_execution:
                run_name = self.xpresso_run_name.split("__")

                if (self.pipeline=='inference'):

                    exp_run = 'm' + run_name[0]
                    run_params = {
                        "parameters_filename": f"config/config_{self.use_case}.json",
                        "parameters_commit_id": None,
                        "training_branch": None,
                        "training_dataset": None,
                        "training_data_version": None,
                        "key_metric": None,
                        "pipeline": self.pipeline,
                        "use_case": self.use_case,
                        "ontology": self.ontology
                    }

                    exp_name, exp_version = "inference", int(constants_config['VERSION']['inference'])
                    trigger_pipeline_run(exp_run, exp_name, exp_version, run_params, self.logger)

                try:
                    if (self.pipeline=='modeling'):
                        solution_name = run_name[1]; pipeline_name = run_name[2]; branch_name = self.xpresso_run_name + "_model"

                        initial_pipeline_rn = json.load(open(f"/project/config/pipeline_execution/run_name_{self.use_case}.json"))
                        unique_run_name = initial_pipeline_rn['xpresso_run_name'].split('__')[0]
                            
                        uid, started_at, metadata_status = get_run_metadata(initial_pipeline_rn['xpresso_run_name'], self.logger)
                        if (metadata_status == 'success'):
                            curr_datetime = datetime.now() + pd.Timedelta(hours=5.5)
                            elapsed_time = round(((curr_datetime - started_at).seconds) / 3600, 3)
                            send_email(solution_name, pipeline_name, self.use_case, unique_run_name, run_name, uid, started_at, elapsed_time, branch_name, self.logger)
                except Exception as my_error:
                    _, _, exc_tb = sys.exc_info()
                    self.logger.error(f'Error {my_error}, Line No {exc_tb.tb_lineno}')

        if not self.local_execution:
            if os.path.exists(f"/pipeline/config/config_{self.use_case}.json"):
                os.remove(f"/pipeline/config/config_{self.use_case}.json")

    def str2bool(self, input_string: str):
        """Convert string to bool  value
        Args:
            input_string(str): string to convert to bool
        Returns:
            bool: True, if string is either true/True else False
        """
        if not input_string:
            return False
        true_set = {"true", "True"}
        return input_string.lower() in true_set

    def on_terminate(self):
        """
        This method is called by the local/remote proxy terminate method
        In remote execution, this would be called when a user terminates the pipeline from the UI
        In local execution, this would be called from a separate thread for testing termination
        Developers must implement any cleanup required in this method
        """
        # === Your termination code goes here ===
        pass

    def on_pause(self, push_exp=True):
        """
        This method is called by the local/remote proxy pause method
        In remote execution, this would be called when a user pauses the pipeline from the UI
        In local execution, this would be called from a separate thread for testing pause
        Developers must implement any cleanup required in this method,
        and save state of the program to disk
        Args:
            push_exp: Whether to push the data present in the output folder
               to the versioning system. This is required once training is
               completed and model needs to be versioned
        """
        # === Your pause code goes here ===
        pass

    def on_restart(self):
        """
        This method is called by the local/remote proxy restart method
        In remote execution, this would be called when a user restarts the pipeline from the UI
        In local execution, this would be called from a separate thread for testing restart
        Developers should implement the logic to
        reload the state of the previous run.
        """
        # === Your restart code goes here ===
        pass



