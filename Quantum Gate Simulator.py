import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector, plot_histogram
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from datetime import datetime

class QuantumSimulatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Quantum Qubit Simulator")
        self.root.geometry("1000x650")
        
        # Theme colors
        self.COLORS = {
            "bg_dark": "#121212", "bg_medium": "#1E1E1E", "bg_light": "#2D2D2D",
            "text": "#E0E0E0", "accent": "#BB86FC", "accent_alt": "#03DAC5",
            "button": "#3700B3", "button_active": "#6200EE"
        }
        self.root.configure(bg=self.COLORS["bg_dark"])
        self._setup_theme()
        
        # Variables
        self.alpha_var = tk.StringVar(value="1")
        self.beta_var = tk.StringVar(value="0")
        self.init_gate_var = tk.StringVar(value="NONE")
        self.update_gate_var = tk.StringVar(value="NONE")
        self.base_circ = None
        self.operations_history = []
        self.state_history = []
        
        self._build_interface()
        plt.style.use('dark_background')
        
    def _setup_theme(self):
        style = ttk.Style()
        style.theme_use('alt')
        
        # Configure styles
        for item, bg, fg in [
            ("TFrame", self.COLORS["bg_dark"], None),
            ("TLabelframe", self.COLORS["bg_dark"], None),
            ("TLabelframe.Label", self.COLORS["bg_dark"], self.COLORS["accent"]),
            ("TLabel", self.COLORS["bg_dark"], self.COLORS["text"]),
            ("TButton", self.COLORS["button"], self.COLORS["text"]),
            ("TEntry", self.COLORS["bg_light"], "#000000"),
            ("TCombobox", self.COLORS["bg_light"], self.COLORS["text"]),
            ("TNotebook", self.COLORS["bg_dark"], None),
            ("TNotebook.Tab", self.COLORS["bg_light"], self.COLORS["text"])
        ]:
            if fg:
                style.configure(item, background=bg, foreground=fg)
            else:
                style.configure(item, background=bg)
                
        style.map("TButton", background=[("active", self.COLORS["button_active"])])
        style.map("TCombobox", fieldbackground=[("readonly", self.COLORS["bg_light"])])
        style.map("TNotebook.Tab", background=[("selected", self.COLORS["accent"])])

    def _build_interface(self):
        # Main layout
        main_frame = ttk.Frame(self.root)
        main_frame.pack(padx=10, pady=10, fill="both", expand=True)
        
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side="left", padx=5, fill="y")
        
        viz_container = ttk.Frame(main_frame)
        viz_container.pack(side="right", padx=5, fill="both", expand=True)
        
        # Initialization panel
        init_frame = ttk.LabelFrame(control_frame, text="Qubit Initialization")
        init_frame.pack(pady=(0, 5), fill="x")
        
        # Form inputs
        gates = ["NONE", "H", "X", "Y", "Z"]
        for i, (label, var, vals) in enumerate([
            ("α (complex):", self.alpha_var, None),
            ("β (complex):", self.beta_var, None),
            ("Initial Gate:", self.init_gate_var, gates)
        ]):
            ttk.Label(init_frame, text=label).grid(row=i, column=0, sticky="w", padx=5, pady=3)
            if vals:
                ttk.Combobox(init_frame, values=vals, textvariable=var, state="readonly", width=13).grid(
                    row=i, column=1, padx=5, pady=3)
            else:
                ttk.Entry(init_frame, textvariable=var, width=15).grid(row=i, column=1, padx=5, pady=3)
        
        ttk.Button(init_frame, text="Initialize & Simulate", command=self.run_simulation).grid(
            row=3, column=0, columnspan=2, padx=5, pady=10, sticky="ew")
        
        # Update panel
        update_frame = ttk.LabelFrame(control_frame, text="Qubit Update")
        update_frame.pack(pady=5, fill="x")
        
        ttk.Label(update_frame, text="Apply Gate:").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        ttk.Combobox(update_frame, values=gates, textvariable=self.update_gate_var, state="readonly", width=13).grid(
            row=0, column=1, padx=5, pady=3)
        ttk.Button(update_frame, text="Apply to Qubit", command=self.apply_update).grid(
            row=1, column=0, columnspan=2, padx=5, pady=10, sticky="ew")
        
        # Report button
        report_frame = ttk.LabelFrame(control_frame, text="Report")
        report_frame.pack(pady=5, fill="x")
        ttk.Button(report_frame, text="Generate Report", command=self.generate_report).pack(
            padx=5, pady=10, fill="x")
        
        # Statistics panel
        self.stats_frame = ttk.LabelFrame(control_frame, text="Statistics")
        self.stats_frame.pack(pady=5, fill="both", expand=True)
        
        # Visualization tabs
        self.viz_notebook = ttk.Notebook(viz_container)
        self.viz_notebook.pack(fill="both", expand=True)
        
        # Create tabs
        self.tabs = {}
        for name in ["Bloch Sphere", "Measurements", "Circuit", "Report"]:
            tab = ttk.Frame(self.viz_notebook)
            self.viz_notebook.add(tab, text=name)
            self.tabs[name.lower().replace(" ", "_")] = tab

        
    def run_simulation(self):
        try:
            # Parse input
            alpha, beta = complex(self.alpha_var.get()), complex(self.beta_var.get())
            if alpha == 0 and beta == 0:
                messagebox.showerror("Invalid State", "Alpha and Beta cannot both be zero.")
                return
                
            # Normalize
            norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
            alpha, beta = alpha/norm, beta/norm
            
            # Reset history
            self.operations_history = []
            self.state_history = []
            
            # Create circuit and apply initial gate
            self.base_circ = QuantumCircuit(1)
            self.base_circ.initialize([alpha, beta], 0)
            
            # Record initial state
            self.operations_history.append(f"Initialized qubit to state: α=({self._format_complex(alpha)})|0⟩ + β=({self._format_complex(beta)})|1⟩")
            self.state_history.append([alpha, beta])
            
            gate = self.init_gate_var.get().upper()
            if gate != "NONE":
                self._apply_gate(gate)
            
            self._simulate_and_display()
            
        except ValueError:
            messagebox.showerror("Input Error", "Invalid complex number input.")

    def _format_complex(self, c):
        return f"{c.real:.4f}{'+' if c.imag >= 0 else ''}{c.imag:.4f}j"

    def _apply_gate(self, gate):
        if gate == "H": self.base_circ.h(0)
        elif gate == "X": self.base_circ.x(0)
        elif gate == "Y": self.base_circ.y(0)
        elif gate == "Z": self.base_circ.z(0)
        
        self.operations_history.append(f"Applied {gate} gate")
        
        # Calculate new state
        state = Statevector.from_instruction(self.base_circ)
        self.state_history.append([state.data[0], state.data[1]])

    def apply_update(self):
        if self.base_circ is None:
            messagebox.showwarning("No Qubit", "Initialize the qubit first.")
            return
            
        gate = self.update_gate_var.get().upper()
        if gate != "NONE":
            self._apply_gate(gate)
            self._simulate_and_display()

    def _simulate_and_display(self):
        # Calculate expected state
        state_after = Statevector.from_instruction(self.base_circ)
        p0e, p1e = abs(state_after.data[0])**2, abs(state_after.data[1])**2
        
        # Build and run measurement circuit
        meas = QuantumCircuit(1, 1)
        meas.compose(self.base_circ, inplace=True)
        meas.measure(0, 0)
        
        sim = AerSimulator()
        res = sim.run(meas, shots=4096).result()
        counts = res.get_counts()
        
        # Calculate statistics
        shots = 4096
        m0, m1 = counts.get('0', 0), counts.get('1', 0)
        p0m, p1m = m0/shots, m1/shots
        
        # Deviations and uncertainties
        d0, d1 = abs(p0e - p0m), abs(p1e - p1m)
        e0, e1 = np.sqrt(p0e * (1 - p0e) / shots), np.sqrt(p1e * (1 - p1e) / shots)
        
        self.display_stats(p0e, p1e, p0m, p1m, d0, d1, e0, e1)
        self.display_visualizations(state_after, counts, meas)

    def display_stats(self, p0e, p1e, p0m, p1m, d0, d1, e0, e1):
        # Clear previous stats
        for widget in self.stats_frame.winfo_children():
            widget.destroy()
            
        # Create stats display
        stats_container = tk.Frame(self.stats_frame, bg=self.COLORS["bg_dark"])
        stats_container.pack(fill="both", expand=True, padx=5, pady=5)
        
        for title, data in [
            ("Theoretical", f"|0⟩: {p0e:.4f}  |1⟩: {p1e:.4f}"),
            ("Measured", f"|0⟩: {p0m:.4f}  |1⟩: {p1m:.4f}"),
            ("Deviation", f"|0⟩: {d0:.4f}, |1⟩: {d1:.4f}"),
            ("Uncertainty", f"|0⟩: ±{e0:.4f}, |1⟩: ±{e1:.4f}")
        ]:
            frame = tk.Frame(stats_container, bg=self.COLORS["bg_medium"], padx=5, pady=5)
            frame.pack(fill="x", pady=(0, 5))
            tk.Label(frame, text=title, bg=self.COLORS["bg_medium"], 
                    fg=self.COLORS["accent_alt"], font=("Helvetica", 9, "bold")).pack(anchor="w")
            tk.Label(frame, text=data, bg=self.COLORS["bg_medium"], 
                    fg=self.COLORS["text"]).pack(anchor="w", padx=10)

    def display_visualizations(self, state, counts, meas):
        # Update plots params
        plt.rcParams.update({
            'figure.facecolor': self.COLORS["bg_dark"],
            'axes.facecolor': self.COLORS["bg_medium"],
            'text.color': self.COLORS["text"],
            'axes.labelcolor': self.COLORS["text"],
            'xtick.color': self.COLORS["text"],
            'ytick.color': self.COLORS["text"]
        })
        
        # Clear previous visualizations
        for tab_name in ["bloch_sphere", "measurements", "circuit"]:
            for widget in self.tabs[tab_name].winfo_children():
                widget.destroy()
        
        # Create visualizations
        figs = [
            (plot_bloch_multivector(state), self.tabs["bloch_sphere"]),
            (plot_histogram(counts, color=[self.COLORS["accent"], self.COLORS["accent_alt"]]), self.tabs["measurements"]),
            (meas.draw('mpl', style={'backgroundcolor': self.COLORS["bg_medium"]}), self.tabs["circuit"])
        ]
        
        # Add to tabs
        for fig, tab in figs:
            canvas = FigureCanvasTkAgg(fig, master=tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

    def generate_report(self):
        if self.base_circ is None:
            messagebox.showwarning("No Qubit", "Initialize the qubit first.")
            return
        
        # Clear previous report
        for widget in self.tabs["report"].winfo_children():
            widget.destroy()
        
        # Create scrolled text area
        report_text = scrolledtext.ScrolledText(self.tabs["report"], bg=self.COLORS["bg_medium"], 
                                               fg=self.COLORS["text"], font=("Consolas", 10))
        report_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Generate report
        state = Statevector.from_instruction(self.base_circ)
        alpha, beta = state.data[0], state.data[1]
        
        # Generate report content
        report = f"""
QUANTUM QUBIT SIMULATION REPORT
{'-'*40}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY OF OPERATIONS:
{'-'*40}
"""
        # Add operation history
        for i, op in enumerate(self.operations_history):
            report += f"{i+1}. {op}\n"
        
        # Add final state
        report += f"""
FINAL QUANTUM STATE:
{'-'*40}
|ψ⟩ = {self._format_complex(alpha)}|0⟩ + {self._format_complex(beta)}|1⟩

Probability of measuring |0⟩: {abs(alpha)**2:.4f}
Probability of measuring |1⟩: {abs(beta)**2:.4f}

MATHEMATICAL BACKGROUND:
{'-'*40}
1. Qubit Representation:
   |ψ⟩ = α|0⟩ + β|1⟩ where |α|² + |β|² = 1

2. Quantum Gates Applied:
"""
        # Add matrices for gates used
        gate_info = {
            "H": """   Hadamard (H): 1/√2 * [1  1]
                [1 -1]
   Effect: Creates superposition""",
            "X": """   Pauli-X (X): [0  1]
                [1  0]
   Effect: Bit flip (NOT gate)""",
            "Y": """   Pauli-Y (Y): [0  -i]
                [i   0]
   Effect: Bit and phase flip""",
            "Z": """   Pauli-Z (Z): [1   0]
                [0  -1]
   Effect: Phase flip"""
        }
        
        gates_used = set([op.split()[1] for op in self.operations_history if "Applied" in op])
        for gate in gates_used:
            if gate in gate_info:
                report += f"{gate_info[gate]}\n\n"
        
        # Add state transformation explanation
        report += """3. State Transformation:
   |ψ'⟩ = U|ψ⟩ - New amplitudes calculated by matrix multiplication
   
4. Measurement:
   Qubit collapses to |0⟩ with p=|α|² or |1⟩ with p=|β|²
"""
        
        # Add state evolution if multiple operations
        if len(self.state_history) > 1:
            report += f"""
STATE EVOLUTION:
{'-'*40}
"""
            for i, (alpha, beta) in enumerate(self.state_history):
                report += f"Step {i}: |ψ⟩ = {self._format_complex(alpha)}|0⟩ + {self._format_complex(beta)}|1⟩\n"
                report += f"       Probabilities: |0⟩: {abs(alpha)**2:.4f}, |1⟩: {abs(beta)**2:.4f}\n\n"
        
        report_text.insert(tk.END, report)
        report_text.configure(state='disabled')  # Make read-only
        self.viz_notebook.select(self.tabs["report"])


if __name__ == '__main__':
    root = tk.Tk()
    app = QuantumSimulatorGUI(root)
    root.mainloop()