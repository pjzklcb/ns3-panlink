# PANLink

PANLink is a project that uses AI/ML algorithms to improve the performance of Wi-Fi MAC protocols for next-generation Wi-Fi networks. The project is based on ns-3 and ns3-ai.

## Features

* Uses AI/ML algorithms to optimize Wi-Fi MAC protocols

## Installation

To install PANLink, follow these steps:

1. **Fetch the ns-3 repository**: `git clone https://gitlab.com/nsnam/ns-3-dev.git`
2. **Navigate to the ns-3 directory**: `cd ns-3-dev`
3. **Integrate PANLink into ns-3**: `git clone https://github.com/pjzklcb/ns3-panlink.git contrib/panlink`

Note: Make sure you have Git installed on your system before running these commands.

## Usage

To use PANLink, follow these steps:

1. **Run the simulation**: `python contrib/panlink/examples/baseline/run_baseline.py --show_log`
2. **View the results**: Check the output files in the folder `result-<timestamp>/`, which will be generated automatically after the simulation is complete.

Note: The `<timestamp>` in the folder name will be replaced with the actual timestamp of the simulation run.
