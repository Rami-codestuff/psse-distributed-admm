import os
import sys
import random
import time
import multiprocessing as mp
import psutil

# --- PSS/E 36.5 PATH INJECTION ---
PSSE_PATH = r"C:\Program Files\PTI\PSSE36\36.5\PSSPY314"
PSSBIN_PATH = r"C:\Program Files\PTI\PSSE36\36.5\PSSBIN"

sys.path.append(PSSE_PATH)
sys.path.append(PSSBIN_PATH)
os.environ['PATH'] = PSSE_PATH + ';' + PSSBIN_PATH + ';' + os.environ['PATH']

# --- CONFIGURATION ---
MAX_ADMM_ITERATIONS = 50
ALPHA = 0.35


# =====================================================================
# TEXT EXPORT FUNCTION
# =====================================================================
def export_results_to_txt(voltages_dict, filename="ADMM_Final_Results.txt"):
    """Generates a clean text file of the final converged network voltages."""
    with open(filename, 'w') as f:
        f.write("===========================================================================\n")
        f.write("          PSS/E ADMM DISTRIBUTED SOLVER - FINAL NETWORK RESULTS\n")
        f.write("===========================================================================\n\n")
        f.write(f"{'Bus ID':<10} | {'Final Voltage (p.u.)':<20}\n")
        f.write("-" * 35 + "\n")

        # Sort the buses numerically so it is easy to read
        for bus in sorted(voltages_dict.keys()):
            f.write(f"{bus:<10} | {voltages_dict[bus]:<20.5f}\n")

        f.write("-" * 35 + "\n")
        f.write("End of Report.\n")
        f.write("===========================================================================\n")


# =====================================================================
# WORKER PROCESS: This runs independently on each CPU Core
# =====================================================================
def worker_init():
    """Initializes PSS/E in the background for each separate CPU core."""
    import psse3605
    import psspy
    psspy.psseinit(50)
    # Silence PSS/E
    psspy.progress_output(6, '', [0, 0])
    psspy.prompt_output(6, '', [0, 0])
    psspy.report_output(6, '', [0, 0])
    psspy.alert_output(6, '', [0, 0])


def solve_zone_local(task_data):
    """The local physics step executed simultaneously by the 4 cores."""
    import psspy
    Z, dummy_targets, dummy_buses, internal_bus_list = task_data

    # Load the specific Zone RAW file
    psspy.read(0, f'Zone_{Z}.raw')

    # Apply the dummy targets requested by the Master
    for dummy_bus in dummy_buses:
        target_v = dummy_targets[dummy_bus]
        psspy.plant_data_4(dummy_bus, 0, realar1=target_v)

    # Solve Local Matrix
    psspy.fnsl([0, 0, 0, 1, 1, 0, 99, 0])

    # Extract calculated internal voltages
    local_voltages = {}
    ierr, (all_buses,) = psspy.abusint(-1, 1, ['NUMBER'])
    ierr, (all_volts,) = psspy.abusreal(-1, 1, ['PU'])

    for b, v in zip(all_buses, all_volts):
        if b in internal_bus_list:
            local_voltages[b] = v

    return Z, local_voltages


# =====================================================================
# MASTER PROCESS: Orchestrates the ADMM Negotiation
# =====================================================================
def run_psse_admm_parallel():
    import psse3605
    import psspy

    print("=" * 75)
    print("   PSS/E TRUE MULTI-CORE ADMM SOLVER (IEEE 30 BUS)")
    print("=" * 75)

    # 1. Load Allocation File
    bus_to_zone = {}
    zone_to_internal = {}
    with open('partition_allocation.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('BusID'): continue
            parts = line.split(',')
            b, z = int(parts[0]), int(parts[1])
            bus_to_zone[b] = z
            if z not in zone_to_internal: zone_to_internal[z] = []
            zone_to_internal[z].append(b)

    num_zones = len(set(bus_to_zone.values()))

    # Map Tie-Lines
    psspy.psseinit(50)
    psspy.progress_output(6, '', [0, 0])
    psspy.prompt_output(6, '', [0, 0])
    psspy.report_output(6, '', [0, 0])
    psspy.alert_output(6, '', [0, 0])

    psspy.read(0, 'IEEE 30 bus.RAW')
    ierr, (from_buses, to_buses) = psspy.abrnint(-1, 1, 1, 1, 1, ['FROMNUMBER', 'TONUMBER'])

    tie_lines = []
    dummy_buses_per_zone = {z: [] for z in range(num_zones)}

    for f, t in zip(from_buses, to_buses):
        z_f, z_t = bus_to_zone[f], bus_to_zone[t]
        if z_f != z_t:
            tie_lines.append((f, t))
            dummy_buses_per_zone[z_f].append(t)
            dummy_buses_per_zone[z_t].append(f)

    # Initialize Dummy Targets
    random.seed(42)
    dummy_setpoints = {z: {dbus: 1.0 + random.uniform(-0.04, 0.04) for dbus in dummy_buses_per_zone[z]} for z in
                       range(num_zones)}

    print(f"-> Orchestrating {num_zones} PSS/E Instances across {mp.cpu_count()} CPU Cores...")
    print("-" * 75)
    header = f"{'Iter':<5} | " + " | ".join([f"Z{z} Err " for z in range(num_zones)]) + " | Status"
    print(header)
    print("-" * 75)

    internal_voltages = {}

    # 2. START TRUE MULTIPROCESSING POOL
    # This boots up 4 separate Python/PSSE processes in the background
    pool = mp.Pool(processes=num_zones, initializer=worker_init)

    for iteration in range(1, MAX_ADMM_ITERATIONS + 1):
        zone_errors = {z: 0.0 for z in range(num_zones)}
        global_max_error = 0.0

        # A. DISPATCH TO 4 CORES SIMULTANEOUSLY
        tasks = []
        for Z in range(num_zones):
            tasks.append((Z, dummy_setpoints[Z], dummy_buses_per_zone[Z], zone_to_internal[Z]))

        # This single line runs all 4 zones at the exact same time
        results = pool.map(solve_zone_local, tasks)

        # Collect results
        for Z, local_v in results:
            internal_voltages.update(local_v)

        # B. CONSENSUS PHASE
        for f, t in tie_lines:
            z_f, z_t = bus_to_zone[f], bus_to_zone[t]

            err_f = abs(dummy_setpoints[z_f][t] - internal_voltages[t])
            err_t = abs(dummy_setpoints[z_t][f] - internal_voltages[f])

            zone_errors[z_f] = max(zone_errors[z_f], err_f)
            zone_errors[z_t] = max(zone_errors[z_t], err_t)
            global_max_error = max(global_max_error, err_f, err_t)

            dummy_setpoints[z_f][t] = (1 - ALPHA) * dummy_setpoints[z_f][t] + ALPHA * internal_voltages[t]
            dummy_setpoints[z_t][f] = (1 - ALPHA) * dummy_setpoints[z_t][f] + ALPHA * internal_voltages[f]

        # C. LOGGING
        err_strings = [f"{zone_errors[z]:<7.5f}" for z in range(num_zones)]
        status = "..." if global_max_error > 1e-4 else "OK!"
        print(f"{iteration:<5} | " + " | ".join(err_strings) + f" | {status}")

        if global_max_error < 1e-4:
            print("-" * 75)
            print(f"CONVERGED! Distributed PSS/E instances synchronized at Iteration {iteration}.")
            break

    # Clean up the 4 background cores
    pool.close()
    pool.join()

    return internal_voltages


if __name__ == '__main__':
    # Fix for Windows Multiprocessing
    mp.freeze_support()

    # 1. Start Timers and True OS-Level Memory Profiler
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    t0_wall = time.perf_counter()

    # 2. RUN TRUE PARALLEL ADMM
    final_voltages = run_psse_admm_parallel()

    # 3. Stop Timers
    admm_wall_time = time.perf_counter() - t0_wall
    mem_after = process.memory_info().rss
    admm_ram_used = (mem_after - mem_before) / (1024 * 1024)  # Convert to MB

    print("\n" + "=" * 75)
    print("   BENCHMARKING: TRUE MULTI-CORE PERFORMANCE")
    print("=" * 75)
    print(f"Total Wall Time (4 Cores): {admm_wall_time * 1000:.3f} ms")
    print(f"True OS RAM Used (MB):     {admm_ram_used:.2f} MB")
    print("===========================================================================\n")

    # 4. EXPORT RESULTS
    export_results_to_txt(final_voltages)
    print(" Saved full solved network results to: ADMM_Final_Results.txt")