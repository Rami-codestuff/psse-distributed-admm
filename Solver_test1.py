import pandas as pd
import numpy as np
import sys


class PowerFlowSolver:
    def __init__(self, filename):
        self.filename = filename
        self.baseMVA = 100.0
        self.bus_data = []
        self.branch_data = []
        self.gen_data = []
        self.load_data = []
        self.shunt_data = []  # NEW: Store capacitor/reactor data

        # System Matrices and State
        self.Ybus = None
        self.V = None  # Complex Voltage Vector
        self.Vm = None  # Voltage Magnitude Vector
        self.Va = None  # Voltage Angle Vector (radians)

        # Mappings
        self.bus_id_to_idx = {}
        self.idx_to_bus_id = {}

    def parse_raw_file(self):
        print(f"Reading {self.filename}...")
        mode = 'HEADER'
        temp_lines = []

        with open(self.filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue

                # Check for section terminators
                if line.startswith('0 /') or line.startswith('Q'):
                    if 'END OF BUS DATA' in line:
                        mode = 'LOAD'
                    elif 'END OF LOAD DATA' in line:
                        mode = 'FIXED SHUNT'
                    elif 'END OF FIXED SHUNT DATA' in line:
                        mode = 'GEN'
                    elif 'END OF GENERATOR DATA' in line:
                        mode = 'BRANCH'
                    elif 'END OF BRANCH DATA' in line:
                        mode = 'TRANSFORMER'
                    elif 'END OF TRANSFORMER DATA' in line:
                        mode = 'FINISHED'
                    continue

                parts = [x.strip("'").strip() for x in line.split(',')]

                if mode == 'HEADER':
                    if 'MVA' in line or len(parts) > 1:
                        try:
                            self.baseMVA = float(parts[1])
                        except:
                            pass
                    mode = 'BUS'

                elif mode == 'BUS':
                    try:
                        bid = int(parts[0])
                    except ValueError:
                        continue
                    self.bus_data.append({
                        'id': bid,
                        'type': int(parts[3]),
                        'Vm': float(parts[7]),
                        'Va': float(parts[8])
                    })

                elif mode == 'LOAD':
                    self.load_data.append({
                        'bus': int(parts[0]),
                        'P': float(parts[5]),
                        'Q': float(parts[6])
                    })

                elif mode == 'FIXED SHUNT':
                    # NEW: Parse Fixed Shunts (Capacitors/Reactors)
                    # Format: Bus, ID, Status, P (MW), Q (MVar)
                    try:
                        bid = int(parts[0])
                        status = int(parts[2])
                        p_shunt = float(parts[3])
                        q_shunt = float(parts[4])

                        if status > 0:
                            self.shunt_data.append({
                                'bus': bid,
                                'G': p_shunt,  # Conductance (MW)
                                'B': q_shunt  # Susceptance (MVar)
                            })
                    except:
                        pass

                elif mode == 'GEN':
                    # Parse Gen Data including Impedance (ZR, ZX)
                    try:
                        zr = float(parts[9])
                        zx = float(parts[10])
                    except IndexError:
                        zr = 0.0
                        zx = 0.001
                    self.gen_data.append({
                        'bus': int(parts[0]),
                        'P': float(parts[2]),
                        'Q': float(parts[3]),
                        'V_set': float(parts[6]),
                        'Zr': zr, 'Zx': zx
                    })

                elif mode == 'BRANCH':
                    self.branch_data.append({
                        'from': int(parts[0]), 'to': int(parts[1]),
                        'r': float(parts[3]), 'x': float(parts[4]), 'b': float(parts[5]),
                        'ratio': 0.0, 'angle': 0.0
                    })

                elif mode == 'TRANSFORMER':
                    try:
                        int(parts[0]);
                        int(parts[1])
                        if len(temp_lines) > 0: self._process_transformer_lines(temp_lines)
                        temp_lines = [parts]
                    except:
                        temp_lines.append(parts)

        if len(temp_lines) > 0: self._process_transformer_lines(temp_lines)

        self.bus_df = pd.DataFrame(self.bus_data).set_index('id')
        for idx, bus_id in enumerate(self.bus_df.index):
            self.bus_id_to_idx[bus_id] = idx
            self.idx_to_bus_id[idx] = bus_id
        print(f"Parsed: {len(self.bus_data)} Buses, {len(self.shunt_data)} Shunts.")

    def _process_transformer_lines(self, lines):
        try:
            f = int(lines[0][0]);
            t = int(lines[0][1])
            r = float(lines[1][0]);
            x = float(lines[1][1])
            ratio = float(lines[2][0])
            self.branch_data.append({
                'from': f, 'to': t, 'r': r, 'x': x, 'b': 0.0,
                'ratio': ratio if ratio > 0 else 1.0, 'angle': 0.0
            })
        except Exception as e:
            print(f"Error parsing transformer: {e}")

    def build_ybus(self):
        print("Building Ybus...")
        nbus = len(self.bus_df)
        self.Ybus = np.zeros((nbus, nbus), dtype=complex)

        # 1. Add Branch Admittances
        for br in self.branch_data:
            try:
                i = self.bus_id_to_idx[br['from']]
                j = self.bus_id_to_idx[br['to']]
                z = complex(br['r'], br['x'])
                y = 1.0 / z
                b_ch = complex(0, br['b'] / 2)
                a = br['ratio']
                if a == 0: a = 1.0

                self.Ybus[i, i] += (y) / (a ** 2) + b_ch
                self.Ybus[j, j] += y + b_ch
                self.Ybus[i, j] -= y / a
                self.Ybus[j, i] -= y / a
            except KeyError:
                pass

        # 2. NEW: Add Fixed Shunts (Capacitors) to Diagonal
        for shunt in self.shunt_data:
            try:
                idx = self.bus_id_to_idx[shunt['bus']]
                # Shunt Admittance Y = (P + jQ) / S_base
                # Note: In PSS/E RAW, Q is positive for capacitor.
                g_pu = shunt['G'] / self.baseMVA
                b_pu = shunt['B'] / self.baseMVA

                # Add to diagonal
                self.Ybus[idx, idx] += complex(g_pu, b_pu)
            except KeyError:
                pass

    def initialize_state(self):
        nbus = len(self.bus_df)
        self.Vm = np.ones(nbus)
        self.Va = np.zeros(nbus)
        self.P_spec = np.zeros(nbus)
        self.Q_spec = np.zeros(nbus)

        for l in self.load_data:
            idx = self.bus_id_to_idx[l['bus']]
            self.P_spec[idx] -= l['P'] / self.baseMVA
            self.Q_spec[idx] -= l['Q'] / self.baseMVA

        for g in self.gen_data:
            idx = self.bus_id_to_idx[g['bus']]
            self.P_spec[idx] += g['P'] / self.baseMVA
            self.Q_spec[idx] += g['Q'] / self.baseMVA
            if self.bus_df.loc[g['bus']]['type'] in [2, 3]:
                self.Vm[idx] = g['V_set']

        self.slack_indices = [self.bus_id_to_idx[b] for b in self.bus_df[self.bus_df['type'] == 3].index]
        self.pv_indices = [self.bus_id_to_idx[b] for b in self.bus_df[self.bus_df['type'] == 2].index]
        self.pq_indices = [self.bus_id_to_idx[b] for b in self.bus_df[self.bus_df['type'] == 1].index]

    def solve_newton_raphson(self, max_iter=20, tol=1e-5):
        print("\nStarting Newton-Raphson Solver...")
        nbus = len(self.bus_df)
        for it in range(max_iter):
            V = self.Vm * np.exp(1j * self.Va)
            I_inj = self.Ybus @ V
            S_calc = V * np.conj(I_inj)
            P_calc = S_calc.real
            Q_calc = S_calc.imag

            dP = self.P_spec - P_calc
            dQ = self.Q_spec - Q_calc

            non_slack = np.array([i not in self.slack_indices for i in range(nbus)])
            pq_only = np.array([i in self.pq_indices for i in range(nbus)])

            mismatch = np.concatenate([dP[non_slack], dQ[pq_only]])
            max_error = np.max(np.abs(mismatch))
            print(f"Iteration {it + 1}: Max Mismatch = {max_error:.6f} p.u.")

            if max_error < tol:
                print("Converged!")
                self.V = V
                return True

            J11 = np.zeros((nbus, nbus));
            J12 = np.zeros((nbus, nbus))
            J21 = np.zeros((nbus, nbus));
            J22 = np.zeros((nbus, nbus))

            for i in range(nbus):
                for k in range(nbus):
                    G = self.Ybus[i, k].real
                    B = self.Ybus[i, k].imag
                    th = self.Va[i] - self.Va[k]
                    if i != k:
                        J11[i, k] = self.Vm[i] * self.Vm[k] * (G * np.sin(th) - B * np.cos(th))
                        J21[i, k] = -self.Vm[i] * self.Vm[k] * (G * np.cos(th) + B * np.sin(th))
                        J12[i, k] = self.Vm[i] * (G * np.cos(th) + B * np.sin(th))
                        J22[i, k] = self.Vm[i] * (G * np.sin(th) - B * np.cos(th))
                    else:
                        J11[i, i] = -Q_calc[i] - (self.Vm[i] ** 2 * B)
                        J21[i, i] = P_calc[i] - (self.Vm[i] ** 2 * G)
                        J12[i, i] = P_calc[i] / self.Vm[i] + (self.Vm[i] * G)
                        J22[i, i] = Q_calc[i] / self.Vm[i] - (self.Vm[i] * B)

            idx_dAng = [i for i in range(nbus) if i not in self.slack_indices]
            idx_dV = [i for i in range(nbus) if i in self.pq_indices]

            J_AA = J11[np.ix_(idx_dAng, idx_dAng)];
            J_AV = J12[np.ix_(idx_dAng, idx_dV)]
            J_VA = J21[np.ix_(idx_dV, idx_dAng)];
            J_VV = J22[np.ix_(idx_dV, idx_dV)]
            J_reduced = np.block([[J_AA, J_AV], [J_VA, J_VV]])

            deltas = np.linalg.solve(J_reduced, mismatch)
            split_idx = len(idx_dAng)
            dAng = deltas[:split_idx];
            dV = deltas[split_idx:]

            for i, bus_idx in enumerate(idx_dAng): self.Va[bus_idx] += dAng[i]
            for i, bus_idx in enumerate(idx_dV): self.Vm[bus_idx] += dV[i]
        return False

    def build_fault_zbus(self):
        print("\nBuilding Fault Impedance Matrix (Zbus)...")
        # Start with system Ybus (Lines + Shunts)
        Ybus_fault = self.Ybus.copy()

        # Add Generator Impedances to Diagonals
        for g in self.gen_data:
            bus_idx = self.bus_id_to_idx[g['bus']]
            z_gen = complex(g['Zr'], g['Zx'])
            if abs(z_gen) < 1e-6: z_gen = complex(0, 0.0001)
            Ybus_fault[bus_idx, bus_idx] += 1.0 / z_gen

        try:
            self.Zbus_fault = np.linalg.inv(Ybus_fault)
            print("Zbus built successfully.")
        except np.linalg.LinAlgError:
            print("Error: Fault Ybus is singular.")

    def run_short_circuit_scan(self):
        if self.V is None:
            print("Running Power Flow first to get Pre-Fault Voltages...")
            if not self.solve_newton_raphson(): return

        if not hasattr(self, 'Zbus_fault'): self.build_fault_zbus()

        print(f"\n{'Bus':<5} {'Pre-Fault V':<12} {'Z_th (pu)':<12} {'I_fault (pu)':<12}")
        print("-" * 45)
        for idx in range(len(self.bus_df)):
            bus_id = self.idx_to_bus_id[idx]
            V_pre = self.V[idx]
            Z_th = self.Zbus_fault[idx, idx]
            I_fault = V_pre / Z_th
            print(f"{bus_id:<5} {abs(V_pre):<12.4f} {abs(Z_th):<12.4f} {abs(I_fault):<12.4f}")

    def apply_fault_at_bus(self, fault_bus_id):
        if self.V is None:
            print("Running Power Flow first...")
            if not self.solve_newton_raphson(): return
        if not hasattr(self, 'Zbus_fault'): self.build_fault_zbus()

        try:
            k = self.bus_id_to_idx[fault_bus_id]
        except KeyError:
            print(f"Error: Bus {fault_bus_id} not found.")
            return

        V_fault_bus_pre = self.V[k]
        Z_kk = self.Zbus_fault[k, k]
        I_fault = V_fault_bus_pre / Z_kk

        print(f"\nFault at Bus {fault_bus_id} | Current: {abs(I_fault):.4f} p.u.")
        print("-" * 55)
        print(f"{'Bus':<5} {'Pre-Fault V':<12} {'Post-Fault V':<12} {'Status'}")
        print("-" * 55)

        for i in range(len(self.bus_df)):
            bus_id = self.idx_to_bus_id[i]
            V_pre = self.V[i]
            Z_ik = self.Zbus_fault[i, k]
            V_post = V_pre - (Z_ik * I_fault)

            status = ""
            if abs(V_post) < 0.001:
                status = "<< FAULT LOC"
            elif abs(V_post) < 0.5:
                status = "* SEVERE SAG *"
            elif abs(V_post) < 0.8:
                status = "* SAG *"
            print(f"{bus_id:<5} {abs(V_pre):<12.4f} {abs(V_post):<12.4f} {status}")

    def save_results(self):
        print("\n--- Final Power Flow Results ---")
        print(f"{'Bus ID':<10} {'Volt (pu)':<10} {'Angle (deg)':<10}")
        print("-" * 35)
        for idx in range(len(self.bus_df)):
            bus_id = self.idx_to_bus_id[idx]
            v_mag = self.Vm[idx]
            v_ang = np.degrees(self.Va[idx])
            print(f"{bus_id:<10} {v_mag:<10.4f} {v_ang:<10.2f}")


if __name__ == "__main__":
    #solver = PowerFlowSolver("IEEE 30 bus.RAW")
    #solver = PowerFlowSolver("IEEE 118 Bus.RAW")
    solver = PowerFlowSolver("IEEE300Bus.raw")
    solver.parse_raw_file()
    solver.build_ybus()
    solver.initialize_state()

    while True:
        print("\n" + "=" * 30)
        print("   POWER SYSTEM SOLVER MENU")
        print("=" * 30)
        print("1. Run Power Flow (Newton-Raphson)")
        print("2. Run Short Circuit Scan (All Buses)")
        print("3. Apply Fault at Specific Bus")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ")

        if choice == '1':
            if solver.solve_newton_raphson():
                solver.save_results()

        elif choice == '2':
            solver.run_short_circuit_scan()

        elif choice == '3':
            try:
                bid = int(input("Enter Bus ID to fault: "))
                solver.apply_fault_at_bus(bid)
            except ValueError:
                print("Invalid input! Please enter a number.")

        elif choice == '4':
            print("Exiting...")
            sys.exit()

        else:
            print("Invalid choice, please try again.")