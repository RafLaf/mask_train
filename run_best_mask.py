import subprocess
import yaml
import logging
import re

def extract_metric(output):
    # Split the output into lines and search for the "Best accuracy" line
    lines = output.splitlines()
    for line in lines:
        if "Best accuracy" in line:
            # Extract and return the number after the colon
            return float(line.split(":")[-1].strip())
    return None


def parse_log_file(log_file):
    """
    Parse the log file to extract configuration details for each num_shots.
    """
    trials = []
    pattern = re.compile(
        r"Best trial: \{'num_shots': (?P<num_shots>\d+), 'best_accuracy': (?P<best_accuracy>[\d.]+), "
        r"'lr_mask': (?P<lr_mask>[\d.eE\-+]+), 'reg_mask': (?P<reg_mask>[\d.eE\-+]+), "
        r"'epoch_mask': (?P<epoch_mask>\d+), 'config_file': '(?P<config_file>.+?)'\}"
    )

    with open(log_file, "r") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                trials.append({
                    'num_shots': int(match.group('num_shots')),
                    'best_accuracy': float(match.group('best_accuracy')),
                    'lr_mask': float(match.group('lr_mask')),
                    'reg_mask': float(match.group('reg_mask')),
                    'epoch_mask': int(match.group('epoch_mask')),
                    'config_file': match.group('config_file'),
                })

    return trials


def update_config(config_file, lr_mask, reg_mask, epoch_mask, num_support):
    """
    Update the configuration file with the specified parameters.
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Update model parameters
    config['MODEL']['LR_MASK'] = lr_mask
    config['MODEL']['REG_MASK'] = reg_mask
    config['MODEL']['EPOCH_MASK'] = epoch_mask

    # Update num_support parameter
    config['DATA']['TEST']['EPISODE_DESCR_CONFIG']['NUM_SUPPORT'] = num_support

    # Save updated config
    with open(config_file, 'w') as f:
        yaml.safe_dump(config, f)


def run_task(config_file):
    """
    Execute the main task with the updated configuration file.
    """
    process = subprocess.run(['python', 'main_one_task.py', '--cfg', config_file, '0'], capture_output=True, text=True)
    print(process.stdout)  # Print the output of the subprocess
    print(process.stderr)  # Print the error of the subprocess
    return process.returncode , extract_metric(process.stdout)


def main():
    log_file = "output_files/logs/mask_mscoco.log"  # Replace with the path to your log file

    # Setup logging
    logging.basicConfig(filename="output_files/logs/task_execution.log", level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Started processing tasks.")

    # Parse the log file to get trials
    trials = parse_log_file(log_file)

    # Process each trial
    for trial in trials:
        logging.info(f"Processing num_shots: {trial['num_shots']}")
        print(f"Processing num_shots: {trial['num_shots']}")

        # Update the configuration file
        update_config(
            config_file=trial['config_file'],
            lr_mask=trial['lr_mask'],
            reg_mask=trial['reg_mask'],
            epoch_mask=trial['epoch_mask'],
            num_support=trial['num_shots']
        )

        # Run the task
        result_code , acc = run_task(trial['config_file'])
        if result_code == 0:
            logging.info(f"Successfully completed task for num_shots: {trial['num_shots']} and accuracy {acc}")
        else:
            logging.error(f"Task failed for num_shots: {trial['num_shots']}")


if __name__ == "__main__":
    main()
