import subprocess
import yaml
import optuna
import argparse
import logging


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
    reg_mask = trial.suggest_float('reg_mask', 1e-5, 1.0, log=True)
    lr_mask = trial.suggest_float('lr_mask', 1e-9, 1.0, log=True)
    epoch_mask = trial.suggest_int('epoch_mask', 1, 100)

    # Load the existing config.yaml
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update the specific hyperparameters in the MODEL section
    config['MODEL']['LR_MASK'] = lr_mask
    config['MODEL']['REG_MASK'] = reg_mask
    config['MODEL']['EPOCH_MASK'] = epoch_mask  

    # Save the updated config.yaml
    with open(config_file, 'w') as f:
        yaml.safe_dump(config, f)

    # Run the experiment via subprocess, passing your command-line arguments
    process = subprocess.run(['python', 'main_one_task.py', '--cfg', config_file, '0'], capture_output=True, text=True)
    
    # Extract the output from stdout and find the best accuracy
    output = process.stdout
    result = extract_metric(output)
    print(f'\n \n Result obtained {result} \n \n with \n lr_mask: {lr_mask} \n reg_mask: {reg_mask} \n epoch_mask: {epoch_mask}')
    return result

# Run the optimization using Optuna
def do_study(config_file, n_trials):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, config_file), n_trials=n_trials)

    # Print the best trial and its hyperparameters
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print(f"  Params: {trial.params}")
    print(f"  Config file: {config_file}")
    return study


def log_best_results(study, config_file, shots):
    # Get the best trial
    best_trial = study.best_trial

    # Create a dictionary with the best trial's parameters and value
    best_data = {
        'num_shots': shots,
        'best_accuracy': best_trial.value,
        'lr_mask': best_trial.params['lr_mask'],
        'reg_mask': best_trial.params['reg_mask'],
        'epoch_mask': best_trial.params['epoch_mask'],
        'config_file': config_file
    }

    # Log the best trial at the end of optimization
    logging.info(f'Best trial: {best_data}')

    print(f"Best trial:\n {best_data}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization for MASK model")
    parser.add_argument('config_file', type=str, help='Path to the config file')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--vary_shots', action='store_true', help='loop though the nu;ber of shots')  # <-- New boolean flag
    parser.add_argument('--n_shots', type=int, default=50, help='number of shots')
    parser.add_argument('--log_file', type=str, help='Path to the config file')

    args = parser.parse_args()

    logging.basicConfig(filename=args.log_file, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Started Optuna optimization')

    def update_config(config_file, num_support):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        print(config['DATA']['TEST']['EPISODE_DESCR_CONFIG']['NUM_SUPPORT'])
        config['DATA']['TEST']['EPISODE_DESCR_CONFIG']['NUM_SUPPORT'] = num_support
        with open(config_file, 'w') as f:
            yaml.safe_dump(config, f)

    if args.vary_shots:
        #for s in list(range(1, 5)) + list(range(5, 70, 5)):
        for s in range(60, 70, 5):
            update_config(args.config_file, s)
            study = do_study(args.config_file, args.n_trials)
            log_best_results(study, args.config_file, s)
    else:
        update_config(args.config_file, args.n_shots)
        study = do_study(args.config_file, args.n_trials)
        log_best_results(study, args.config_file , args.n_shots)

if __name__ == "__main__":
    main()
