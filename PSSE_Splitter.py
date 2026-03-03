import os
import sys

# --- PSS/E 36.5 PATH INJECTION ---
PSSE_PATH = r"C:\Program Files\PTI\PSSE36\36.5\PSSPY314"
PSSBIN_PATH = r"C:\Program Files\PTI\PSSE36\36.5\PSSBIN"

sys.path.append(PSSE_PATH)
sys.path.append(PSSBIN_PATH)
os.environ['PATH'] = PSSE_PATH + ';' + PSSBIN_PATH + ';' + os.environ['PATH']

# Initialize PSS/E 36.5 environment
import psse3605
import psspy
import redirect


def split_psse_case():
    redirect.psse2py()

    print("=" * 60)
    print("   PSS/E IEEE 30-BUS DISTRIBUTED CASE SPLITTER")
    print("=" * 60)

    # 1. Parse your NEW Allocation File dynamically
    bus_to_zone = {}
    zone_to_bus = {}

    with open('partition_allocation.txt', 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comment headers and empty lines
            if not line or line.startswith('#') or line.startswith('BusID'):
                continue

            # Split the comma-separated data (e.g., "1,2")
            parts = line.split(',')
            bus_id = int(parts[0])
            zone_id = int(parts[1])

            bus_to_zone[bus_id] = zone_id
            if zone_id not in zone_to_bus:
                zone_to_bus[zone_id] = []
            zone_to_bus[zone_id].append(bus_id)

    num_zones = len(zone_to_bus)
    print(f"-> Successfully loaded allocation for {len(bus_to_zone)} buses across {num_zones} zones.")

    # 2. Initialize PSS/E and load 30-Bus master case
    psspy.psseinit(50)
    psspy.read(0, 'IEEE 30 bus.RAW')

    # Extract all branches
    ierr, (from_buses, to_buses) = psspy.abrnint(-1, 1, 1, 1, 1, ['FROMNUMBER', 'TONUMBER'])
    ierr, (ckt_ids,) = psspy.abrnchar(-1, 1, 1, 1, 1, ['ID'])
    all_branches = list(zip(from_buses, to_buses, ckt_ids))

    # Find Tie-Lines
    tie_lines = []
    for f, t, ckt in all_branches:
        if bus_to_zone[f] != bus_to_zone[t]:
            tie_lines.append((f, t, ckt))

    print(f"-> Identified {len(tie_lines)} active Tie-Lines.")
    print("-" * 60)

    # 3. Perform the Matrix Surgery dynamically for however many zones exist
    for Z in sorted(zone_to_bus.keys()):
        print(f"\n>>> Extracting Sub-system for Zone {Z} <<<")

        # Reload fresh master case
        psspy.read(0, 'IEEE 30 bus.RAW')

        internal = zone_to_bus[Z]
        dummies = []

        for f, t, ckt in tie_lines:
            if f in internal and t not in internal:
                dummies.append(t)
            elif t in internal and f not in internal:
                dummies.append(f)

        dummies = list(set(dummies))
        print(f"   Internal Buses: {len(internal)}")
        print(f"   Boundary (Dummy) Buses: {len(dummies)}")

        # A. Sever connections outside the zone
        for f, t, ckt in all_branches:
            if f not in internal and t not in internal:
                psspy.purgbrn(f, t, ckt)

        # B. Convert Boundary Buses into "Virtual Slacks"
        for dbus in dummies:
            psspy.plant_data_4(dbus, 0)
            psspy.machine_data_4(dbus, 'D1', intgar1=1, realar1=1.0)
            psspy.bus_chng_4(dbus, 0, intgar1=3)

        # C. Define Subsystem and Export
        keep_buses = internal + dummies
        ierr = psspy.bsys(1, 0, [0.0, 0.0], 0, [], len(keep_buses), keep_buses, 0, [], 0, [])

        out_file = f'Zone_{Z}.raw'
        psspy.rawd_2(1, 1, [1, 1, 1, 0, 0, 0, 0], 0, out_file)
        print(f"   [SUCCESS] Saved: {out_file}")

    print("\n" + "=" * 60)
    print(f"All {num_zones} zones successfully split for PSS/E.")


if __name__ == '__main__':
    split_psse_case()