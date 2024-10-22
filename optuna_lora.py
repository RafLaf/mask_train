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
    lr_lora = trial.suggest_float('lr_lora', 1e-5, 1e-1, log=True)
    reg_lora = trial.suggest_float('regularization_lora', 1e-9, 1.0, log=True)
    epoch_lora = trial.suggest_int('epoch_lora', 1, 100)
    config_file = 'configs/lora/mscoco_DINO_v2_scikit_lora.yaml'

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
    #print('########## showing output')
    #print(output)
    #print(process.stderr)
    print(output)
    result = extract_metric(output)
    print(f'\n \n Result obtained {result} \n \n with \n lr_lora: {lr_lora} \n reg_lora: {reg_lora} \n epoch_lora: {epoch_lora}')
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
