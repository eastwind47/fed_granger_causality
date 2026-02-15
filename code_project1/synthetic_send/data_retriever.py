# Importing all the necessary packages
import configparser
import os
import pandas as pd
import numpy as np

class RetrieveData:
    def __init__(self, ini_file_name):
        """
        Initialize the class with configuration file name
        """

        
        # config.ini file name
        self.ini_file_name              = ini_file_name

        # Loading the configuration file
        self.config                     = configparser.ConfigParser()
        self.config.read(self.ini_file_name)

        # Number of components
        self.num_components             = int(self.config['COMPONENTS']['num_components'])

        # Total time for the time series
        self.total_time                 = int(self.config['COMPONENTS']['total_time'])

        # Global learning rate
        self.global_learning_rate       = float(eval(self.config['GLOBAL_PARAMETERS']['gamma_g']))

        # Global regularization rate
        self.global_regularization_rate = float(eval(self.config['GLOBAL_PARAMETERS']['lambda_g']))

        # # Proportion of training data
        self.training_prop              = float(eval(self.config['TRAINING']['train_set_prop']))

        # Training time
        self.training_time              = int(np.ceil(self.training_prop * self.total_time))

        # Validation time (optional; if omitted, infer from validation Y.csv length)
        if self.config.has_option('VALIDATION', 'validation_time'):
            self.validation_time        = int(self.config['VALIDATION']['validation_time'])
        else:
            self.validation_time        = None

        # Initializing data from each component
        self.local_learners_pack        = None

        # Initializing diagonal matrix data for the global learner
        self.global_learners_pack       = None

        # Initializing data for the HMatrix
        self.HMatrix_pack               = None

        # Validation pack
        self.validation_pack            = None

        # Initializing vector that stores size of states in each component
        self.comp_size_vec              = None

        # Size of the global state vector
        self.global_state_size          = None

        # Output size vector 
        self.output_size_vec            = None

        # Global output size
        self.global_output_size         = None

        # Output data location 
        self.results_location            = self.config['LOCATION']['results_location'].strip()

        # Validation results location 
        valid_location_cfg              = self.config['LOCATION']['valid_location'].strip()
        self.valid_location             = valid_location_cfg if valid_location_cfg else os.path.join(self.results_location, 'validation_results')

        # Centralized Kalman Filter pack
        self.CKF_pack                   = None

        # Global Y
        self.global_Y                   = None

        # Global Y valid   
        self.global_Y_valid             = None

        # Output start indices
        self.output_start_vec                = np.zeros((self.num_components, 1), dtype = int)
        # Preparing the data
        self.PrepareData()


    def PrepareData(self):
        """
        Get the data for each component.
        
        Outputs:
        - M: Number of components
        - T: Length of time series data
        - components: A nested dictionary containing properties of all components and the corresponding observation time series
        """

        try:
            # Getting the data folder location from config file
            data_location           = self.config['LOCATION']['data_location'].strip()
            valid_data_location     = self.config['LOCATION']['valid_data_location'].strip()
            if not data_location:
                raise ValueError("`data_location` is empty in [LOCATION].")
            if not valid_data_location:
                raise ValueError("`valid_data_location` is empty in [LOCATION].")
            if not self.results_location:
                raise ValueError("`results_location` is empty in [LOCATION].")

            # Getting the learning rate of local loss
            eta_l_str       = self.config['LOCAL_PARAMETERS']['eta_l']
            eta_l           = [float(x) for x in eta_l_str.split(',')]

            # Getting the learning rate of global loss for each component
            eta_g_str       = self.config['LOCAL_PARAMETERS']['eta_g']
            eta_g           = [float(x) for x in eta_g_str.split(',')]

            # Getting the local regularization parameter
            lambda_l_str    = self.config['LOCAL_PARAMETERS']['lambda_l']
            lambda_l        = [float(x) for x in lambda_l_str.split(',')]

            # Raising an exception if the learning rate and regularization parameter vectors are not the correct size
            if len(eta_l) != self.num_components or len(lambda_l) != self.num_components or len(eta_g) != self.num_components:
                raise ValueError(f"Length of eta_l vector (= {len(eta_l)}) and/or length of lambda_l vector (= {len(lambda_l)}) \
                                 and/or length of eta_g vector (= {len(eta_g)}) != number of components (= {self.num_components})")

            # Dictionary to store all component data and global learner package
            self.local_learners_pack    = {}
            self.global_learners_pack   = {}
            self.HMatrix_pack           = {}
            self.CKF_pack               = {}
            self.validation_pack        = {}
            self.global_Y               = {}
            self.global_Y_valid         = {}
            A_mm                        = {}
            # Initializing total size of state vector
            self.comp_size_vec          = np.zeros((self.num_components, 1), dtype = int)
            self.global_state_size      = 0

            self.output_size_vec        = np.zeros((self.num_components, 1), dtype = int)
            self.global_output_size     = 0

            A_complete_path             = os.path.join(data_location, 'A_complete.csv')
            C_complete_path             = os.path.join(data_location, 'C_complete.csv')
            df_A_complete               = pd.read_csv(A_complete_path, header = None)
            df_C_complete               = pd.read_csv(C_complete_path, header = None)
            A_complete                  = df_A_complete.to_numpy()
            C_complete                  = df_C_complete.to_numpy()
            
            A_complete_valid_path       = os.path.join(valid_data_location, 'A_complete.csv')
            df_A_complete_valid         = pd.read_csv(A_complete_valid_path, header = None)
            A_complete_valid            = df_A_complete_valid.to_numpy()

            for m in range(self.num_components):
                A_matrix_path = os.path.join(data_location, f'C{m+1}/A.csv')
                C_matrix_path = os.path.join(data_location, f'C{m+1}/C.csv')
                Y_path = os.path.join(data_location, f'C{m+1}/Y.csv')
                df_A = pd.read_csv(A_matrix_path, header=None)
                df_C = pd.read_csv(C_matrix_path, header=None)
                df_Y = pd.read_csv(Y_path, header=None)

                A = df_A.to_numpy()
                C = df_C.to_numpy()
                Y = df_Y.to_numpy().T

                if Y.shape[1] != self.total_time: 
                    raise ValueError(f"Length of time series from config file = {self.total_time} != Length of observation series {Y.shape[1]}")
                
                Y_train         = Y[:,:self.training_time]
                # Y_valid         = Y[:,self.training_time:]

                Y_valid_path    = os.path.join(valid_data_location, f'C{m+1}/Y.csv')
                df_Y_valid      = pd.read_csv(Y_valid_path, header = None)

                Y_valid         = df_Y_valid.to_numpy().T

                if self.validation_time is None:
                    self.validation_time = Y_valid.shape[1]
                elif Y_valid.shape[1] != self.validation_time:
                    raise ValueError(
                        f"Validation length mismatch: expected {self.validation_time}, got {Y_valid.shape[1]} for C{m+1}/Y.csv"
                    )
                

                d_m, p_m = C.shape

                self.comp_size_vec[m, 0]        = p_m
                self.global_state_size          += p_m

                # Taking output size
                self.output_size_vec[m, 0]      = d_m
                self.global_output_size         += d_m

                # Initial values and noise parameters for the local model
                B       = np.zeros((p_m, p_m))
                Q       = 0.0005 * np.eye(p_m)
                R       = 0.0005 * np.eye(d_m)
                P0      = Q
                x0      = np.zeros((p_m, 1))

                self.local_learners_pack[f'comp_{m+1}'] = {

                    'B': B,
                    'Q': Q,
                    'R': R,
                    'P0': P0,
                    'x0': x0,
                    'A': A,
                    'C': C,
                    'Y': Y_train,
                    'd_m': d_m,
                    'p_m': p_m,
                    'eta_l': eta_l[m],
                    'eta_g': eta_g[m],
                    'lambda_l': lambda_l[m],
                }

                self.validation_pack[f'comp_{m+1}']     = {
                    'B': B,
                    'Q': Q,
                    'R': R,
                    'P0': P0,
                    'x0': x0,
                    'A': A,
                    'C': C,
                    'Y': Y_valid,
                    'd_m': d_m,
                    'p_m': p_m,
                    'eta_l': eta_l[m],
                    'eta_g': eta_g[m],
                    'lambda_l': lambda_l[m],
                }
                self.global_Y[f'{m+1}']                 = Y
                self.global_Y_valid[f'{m+1}']           = Y_valid
                A_mm[f'{m+1}{m+1}']                             = A

            if self.validation_time is None:
                raise ValueError("Unable to infer validation horizon because no validation series were loaded.")

            global_Y                = np.zeros((self.global_output_size, self.total_time))
            global_Y_valid          = np.zeros((self.global_output_size, self.validation_time))
            d_start_idx             = 0
            for m in range(self.num_components):
                global_Y[d_start_idx:d_start_idx + self.output_size_vec[m, 0], :] = self.global_Y[f'{m+1}']
                global_Y_valid[d_start_idx:d_start_idx + self.output_size_vec[m, 0], :] = self.global_Y_valid[f'{m+1}']
                if m > 0:
                    self.output_start_vec[m, 0]      = d_start_idx
                    
                d_start_idx         = d_start_idx + self.output_size_vec[m, 0]

            self.global_learners_pack['A_mm']               = A_mm
            self.global_learners_pack['gamma_g']            = self.global_learning_rate
            self.global_learners_pack['lambda_g']           = self.global_regularization_rate

            self.HMatrix_pack['comp_data']                  = self.local_learners_pack
            self.HMatrix_pack['gamma_g']                    = self.global_learning_rate
            self.HMatrix_pack['lambda_g']                   = self.global_regularization_rate
            self.HMatrix_pack['p']                          = self.global_state_size
            self.HMatrix_pack['p_vec']                      = self.comp_size_vec
            self.HMatrix_pack['d']                          = self.global_output_size
            self.HMatrix_pack['d_vec']                      = self.output_size_vec     
            self.HMatrix_pack['M']                          = self.num_components
            self.HMatrix_pack['total_time']                 = self.total_time

            self.CKF_pack['A_complete']                     = A_complete
            self.CKF_pack['C_complete']                     = C_complete
            self.CKF_pack['Y']                              = global_Y
            self.CKF_pack['Y_valid']                        = global_Y_valid
            self.CKF_pack['A_complete_valid']               = A_complete_valid

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing required data file: {e.filename}") from e

        except ValueError as e:
            raise ValueError(f"Configuration/data validation error: {e}") from e








        
