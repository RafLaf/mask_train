import subprocess
import yaml
import optuna

def extract_metric(output):
    # Split the output into lines and search for the "Best accuracy" line
    lines = output.splitlines()
    for line in lines:
        if "Best accuracy" in line:
            # Extract and return the number after the colon
            return float(line.split(":")[-1].strip())
    return None

def objective(trial):
    # Suggest hyperparameters to be optimized
    reg_mask= trial.suggest_float('reg_mask', 1e-5, 1.0, log=True)
    lr_mask = trial.suggest_float('lr_mask', 1e-9, 1.0, log=True)
    #reg_mask = trial.suggest_float('regularization_mask', 1e-9, 1.0, log=True)
    epoch_mask = trial.suggest_int('epoch_mask', 1, 100)
    config_file = 'configs/mask/mask_mscoco_50_shots.yaml'

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
    #print('########## showing output')
    #print(output)
    #print(process.stderr)
    result = extract_metric(output)
    print(f'\n \n Result obtained {result} \n \n with \n lr_mask: {lr_mask}  \n epoch_mask: {epoch_mask} \n reg_mask: {reg_mask}')
    return result

# Run the optimization using Optuna
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    # Print the best trial and its hyperparameters
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print(f"  Params: {trial.params}")
