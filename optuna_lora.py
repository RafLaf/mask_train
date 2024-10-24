import subprocess
import yaml
import optuna
import argparse
import logging
import numpy as np

def extract_metric(output):
    # Split the output into lines and search for the "Best accuracy" line
    lines = output.splitlines()
    for line in lines:
        if "Best accuracy" in line:
            # Extract and return the number after the colon
            return float(line.split(":")[-1].strip())
    return None

def objective(trial, config_file):
    # Suggest hyperparameters to be optimized
    lr_lora = trial.suggest_float('lr_lora', 1e-5, 1e-1, log=True)
    reg_lora = trial.suggest_float('regularization_lora', 1e-9, 1.0, log=True)
    epoch_lora = trial.suggest_int('epoch_lora', 1, 100)

    # Load the existing config.yaml
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update the specific hyperparameters in the MODEL section
    config['MODEL']['LR_LORA'] = lr_lora
    config['MODEL']['REGULARIZATION_LORA'] = reg_lora
    config['MODEL']['EPOCH_LORA'] = epoch_lora   

    # Save the updated config.yaml
    with open(config_file, 'w') as f:
        yaml.safe_dump(config, f)

    # Run the experiment via subprocess, passing your command-line arguments
    process = subprocess.run(['python', 'main_one_task.py', '--cfg', config_file, '0'], capture_output=True, text=True)
    
    # Extract the output from stdout and find the best accuracy
    output = process.stdout
    result = extract_metric(output)
    return result

def update_config(config_file, num_support):
    # Load the existing config.yaml
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update the number of support in the appropriate section
    config['DATA']['TEST']['EPISODE_DESCR_CONFIG']['NUM_SUPPORT'] = num_support
    
    # Save the updated config.yaml
    with open(config_file, 'w') as f:
        yaml.safe_dump(config, f)


def log_best_results(study, config_file, shots):
    # Get the best trial
    best_trial = study.best_trial

    # Create a dictionary with the best trial's parameters and value
    best_data = {
        'num_shots': shots,
        'best_accuracy': best_trial.value,
        'lr_lora': best_trial.params['lr_lora'],
        'reg_lora': best_trial.params['regularization_lora'],
        'epoch_lora': best_trial.params['epoch_lora'],
        'config_file': config_file
    }

    # Log the best trial at the end of optimization
    logging.info(f'Best trial: {best_data}')

    print(f"Best trial:\n {best_data}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization for LoRA model")
    parser.add_argument('config_file', type=str, help='Path to the config file')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--vary_shots', action='store_true', help='Vary the number of shots from 5 to 65 in increments of 5')
    parser.add_argument('--n_shots', type=int, default=50, help='Number of shots to use in testing')
    parser.add_argument('--log_file', type=str, help='Path to the log file')

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(filename=args.log_file, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Started Optuna optimization')

    if args.vary_shots:
        #for s in range(5, 70, 5):
        for s in range(1, 5):
            update_config(args.config_file, s)
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: objective(trial, args.config_file), n_trials=args.n_trials)
            
            # Log best results
            best_trial = study.best_trial
            logging.info(f'Best trial for {s} shots: Value: {best_trial.value}, Params: {best_trial.params}')
            print(f'Best trial for {s} shots:\n Value: {best_trial.value}\n Params: {best_trial.params}')
            log_best_results(study, args.config_file, s)
    else:
        update_config(args.config_file, args.n_shots)
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, args.config_file), n_trials=args.n_trials)
        
        # Log best results
        best_trial = study.best_trial
        logging.info(f'Best trial: Value: {best_trial.value}, Params: {best_trial.params}')
        print(f'Best trial:\n Value: {best_trial.value}\n Params: {best_trial.params}')
        log_best_results(study, args.config_file, args.n_shots)

if __name__ == "__main__":
    main()