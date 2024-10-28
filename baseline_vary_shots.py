import subprocess
import yaml
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

def update_config(config_file, num_support):
    # Load the existing config.yaml
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update the number of support in the appropriate section
    config['DATA']['TEST']['EPISODE_DESCR_CONFIG']['NUM_SUPPORT'] = num_support
    
    # Save the updated config.yaml
    with open(config_file, 'w') as f:
        yaml.safe_dump(config, f)

def log_results(config_file, shots):
    # Run the experiment via subprocess, passing your command-line arguments
    process = subprocess.run(['python', 'main_one_task.py', '--cfg', config_file, '0'], capture_output=True, text=True)
    
    # Extract the output from stdout and find the best accuracy
    output = process.stdout
    result = extract_metric(output)

    # Log the results
    logging.info(f'Number of shots: {shots}, Best accuracy: {result}')
    print(f"Number of shots: {shots}\n Best accuracy: {result}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Experiment logging for LoRA model")
    parser.add_argument('config_file', type=str, help='Path to the config file')
    parser.add_argument('--vary_shots', action='store_true', help='Vary the number of shots from 1 to 65')
    parser.add_argument('--n_shots', type=int, default=50, help='Number of shots to use in testing')
    parser.add_argument('--log_file', type=str, help='Path to the log file')

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(filename=args.log_file, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Started logging experiment results')

    if args.vary_shots:
        # Loop through both ranges for shot values
        for s in list(range(1, 5)) + list(range(5, 70, 5)):
            update_config(args.config_file, s)
            log_results(args.config_file, s)
    else:
        update_config(args.config_file, args.n_shots)
        log_results(args.config_file, args.n_shots)

if __name__ == "__main__":
    main()
