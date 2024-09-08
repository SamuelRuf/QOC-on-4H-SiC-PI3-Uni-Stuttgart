import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
import os
import itertools
import time
from quocslib.utils.inputoutput import readjson
from quocslib.Optimizer import Optimizer
from quocslib.utils.AbstractFoM import AbstractFoM
from matplotlib.colors import SymLogNorm
from scipy.interpolate import interp1d

mittelblau = (0/255, 81/255, 158/255)
hellblau = (0/255, 190/255, 255/255)
anthrazit = (62/255, 68/255, 76/255)

# set a parameter to see animations in line
from matplotlib import rc
rc('animation', html='jshtml')

class Spin():
    def __init__(self, spin, gyromagnetic_ratio, id="No id"):
        """
        Initialize a Spin object.

        Parameters:
        - spin: The spin quantum number.
        - gyromagnetic_ratio: The gyromagnetic ratio.
        - id: The identifier for the spin (optional).

        Returns:
        None
        """
        self.spin = spin
        self.gamma = gyromagnetic_ratio
        self.spin_operator = qt.jmat(spin)
        self.sx, self.sy, self.sz = self.spin_operator
        self.id = id
        self.dim = 2 * spin + 1
        self.order_in_System = None # Electron should be 0, nuclei should be 1, 2, 3, ...
        self.coupling_tensors = {}  # A dictionary of couplings with other spins

    def couple(self, other_spin, coupling_tensor):
        """
        Add a coupling tensor to the spin.

        Parameters:
        - spin: The spin to couple to.
        - coupling_constant: The coupling constant.

        Returns:
        None
        """
        self.coupling_tensors[other_spin.id] = coupling_tensor

class SpinSystem():
    def __init__(self, spins):
        """
        Initialize a SpinSystem object. It contains a dictionary of Spin objects using their id as key
        and orders them in the order they were added, for dimensional tensor product purposes.

        Parameters:
        - spins: A list of Spin objects.

        Returns:
        - SpinSystem: A dictionary of Spin objects using the id as a key.
        """
        self.spins = {}
        for spin in spins:
            self.spins[spin.id] = spin
            spin.order_in_system = len(self.spins) - 1
            

class Hamiltonian():
    def __init__(self, D, B, theta, phi, spins, diagonal_detuning=False):
        """
        Initialize a Hamiltonian object.

        Parameters:
        - D: The zero field splitting parameter.
        - B: The magnetic field strength.
        - theta: The polar angle of the magnetic field.
        - phi: The azimuthal angle of the magnetic field.
        - spins: A dict of Spin objects.
        - diagonal_detuning: A boolean indicating whether to consider diagonal detuning (default is False).

        Returns:
        None
        """
        self.D = D  # Zero field splitting
        self.B = B  # B field
        self.theta = theta
        self.phi = phi
        self.spins = spins
        self.diagonal_detuning = diagonal_detuning
        self.base_hamiltonian = self.calc_base_hamiltonian()
        self.base_hamiltonian_rwa = self.calc_base_hamiltonian(rwa=True)
        self.H = self.base_hamiltonian
        self.H_rwa = self.base_hamiltonian_rwa
        self.get_transition_energies()
        self.get_transition_energies_rwa()

    def calc_base_hamiltonian(self, rwa=False):
        """
        Calculate the base Hamiltonian.
        This method calculates the total Hamiltonian by summing up the contributions from different terms.
        The terms included in the calculation are:
        - Zero field splitting Hamiltonian
        - Zeeman splitting Hamiltonian
        - Nuclear coupling Hamiltonian

        Args:
        - rwa (bool): Whether to consider the rotating wave approximation (default: False)

        Returns:
        The total Hamiltonian.
        """
        H = 0
        H += self.calc_zero_field_splitting_hamiltonian()
        H += self.calc_zeeman_splitting_hamiltonian()
        H += self.calc_nuclear_coupling_hamiltonian(rwa=rwa) 
        return H
    
    def calc_zero_field_splitting_hamiltonian(self):
        """
        Calculate the zero field splitting Hamiltonian.
        This method calculates the zero field splitting Hamiltonian based on the electron's spin and spin projection.

        Returns:
        The zero field splitting Hamiltonian.
        """
        sz_e = self.spins["electron"].sz
        s = self.spins["electron"].spin
        H_zero_field = self.D * (sz_e**2 - 1/3 * s * (s + 1))
        H_zero_field = self.embed_hamiltonian(H_zero_field, 0)
        return H_zero_field
        
    def calc_zeeman_splitting_hamiltonian(self):
        """
        Calculate the Zeeman splitting Hamiltonian.

        Returns:
        The Zeeman splitting Hamiltonian.
        """
        H = 0
        for spin in self.spins:
            zeeman_splitting = self.spins[spin].gamma * self.B * (np.sin(self.theta) * np.cos(self.phi) * self.spins[spin].sx + np.sin(self.theta) * np.sin(self.phi) * self.spins[spin].sy + np.cos(self.theta) * self.spins[spin].sz)
            zeeman_splitting = self.embed_hamiltonian(zeeman_splitting, self.spins[spin].order_in_system)
            H += zeeman_splitting
        return H
    
    def calc_nuclear_coupling_hamiltonian(self, rwa=False):
        """
        Calculate the nuclear coupling Hamiltonian using the coupling tensors.

        Args:
        - rwa (bool): Whether to consider the rotating wave approximation. If true sets all couplings except Azz to 0. (default: False)

        Returns:
        The nuclear coupling Hamiltonian.
        """
        H = 0
        for spin in self.spins:
            for other_spin in self.spins[spin].coupling_tensors:
                #if rwa is True, every coupling is made 0 except for the Azz coupling
                if rwa:
                    if self.diagonal_detuning:
                        A = self.spins[spin].coupling_tensors[other_spin]*[[1,0,0],[0,1,0],[0,0,1]]
                    else:
                        A = self.spins[spin].coupling_tensors[other_spin]*[[0,0,0],[0,0,0],[0,0,1]]
                else:
                    A = self.spins[spin].coupling_tensors[other_spin]
                # The dot prod makes Axx*Ix + Axy*Iy + Axz*Iz 
                AxI = np.dot(A[0], self.spins[other_spin].spin_operator)
                AyI = np.dot(A[1], self.spins[other_spin].spin_operator)
                AzI = np.dot(A[2], self.spins[other_spin].spin_operator)
                # insert sx, sy, sz with the ix, iy, iz at the right position
                order_spin = self.spins[spin].order_in_system
                order_other_spin = self.spins[other_spin].order_in_system
                SxAxI_op = [self.spins[spin].sx if self.spins[i].order_in_system == order_spin else AxI if self.spins[i].order_in_system == order_other_spin else qt.qeye(self.spins[i].dim) for i in self.spins]
                SyAyI_op = [self.spins[spin].sy if self.spins[i].order_in_system == order_spin else AyI if self.spins[i].order_in_system == order_other_spin else qt.qeye(self.spins[i].dim) for i in self.spins]
                SzAzI_op = [self.spins[spin].sz if self.spins[i].order_in_system == order_spin else AzI if self.spins[i].order_in_system == order_other_spin else qt.qeye(self.spins[i].dim) for i in self.spins]
                H += qt.tensor(SxAxI_op) + qt.tensor(SyAyI_op) + qt.tensor(SzAzI_op)

        return H
    
    def embed_hamiltonian(self, H, spin_index):
        """
        Embed the Hamiltonian in the dimensions of the other spins.

        Parameters:
        - H: The Hamiltonian to embed.
        - spin_index: The index of the spin to embed the Hamiltonian in.

        Returns:
        The embedded Hamiltonian.
        """
        #make a list with the identity operators for the spins - depending on their dim - and the hamiltonian at the index
        operators = [qt.qeye(self.spins[spin].dim) if self.spins[spin].order_in_system != spin_index else H for spin in self.spins]
        return qt.tensor(operators)
    
    def calc_hamiltonian(self, t, B1=0, omega=0, phi=0, RWA=False):
        """
        Calculate the total Hamiltonian.

        Parameters:
        - B1: The microwave field strength.
        - omega: The microwave frequency.
        - phi: The microwave phase.
        - t: The time.
        - RWA: Whether to use the rotating wave approximation.

        Returns:
        The total Hamiltonian.
        """
        if RWA:
            H = self.base_hamiltonian_rwa.copy()
        else:
            H = self.base_hamiltonian.copy()
        if B1 != 0:
            if not RWA:
                H += self.calc_microwave_hamiltonian(t, B1, omega, phi)
            if RWA:
                H += self.calc_microwave_hamiltonian_RWA(B1, omega, phi)
        return H
    
    def calc_microwave_hamiltonian(self, t, B1, omega, phi=0, mw_theta=np.pi/2, mw_phi=0):
        """
        Calculate the microwave Hamiltonian for different microwave amplitudes.

        Parameters:
        - t: The time.
        - B1: The microwave field strength.
            - Can be a float or a list of the form [B1_amplitudes, B1_timegrid].
        - omega: The microwave frequency.
        - phi: The microwave phase.
        - mw_theta: The polar angle of the microwave field.
        - mw_phi: The azimuthal angle of the microwave field.

        Returns:
        The microwave Hamiltonian.
        """
        if isinstance(B1, list):
            B1_amplitudes = B1[0]
            B1_timegrid = B1[1]
            # if t is between two values of the timegrid, take the amplitude of the first value
            # get the index of the biggest time in the timegrid that is smaller than t
            index = np.searchsorted(B1_timegrid, t, side='right') - 1
            B1_amplitude = B1_amplitudes[index]
        else:
            B1_amplitude = B1
        
        H = 0
        H += B1_amplitude * self.microwave_hamiltonian(t, omega, phi, mw_theta, mw_phi)
        return H
    
    def microwave_hamiltonian(self, t, omega, phi=0, mw_theta=np.pi/2, mw_phi=0):
        """
        Calculate the microwave Hamiltonian normalized to a microwave amplitude of 1.

        Parameters:
        - B1: The microwave field strength.
        - omega: The microwave frequency.
        - phi: The microwave phase.
        - mw_theta: The polar angle of the microwave field.
        - mw_phi: The azimuthal angle of the microwave field.

        Returns:
        The microwave Hamiltonian.
        """
        H = 0
        for spin in self.spins:
            microwave = self.spins[spin].gamma * np.cos(omega * t * 2 * np.pi + phi) * (np.sin(mw_theta) * np.cos(mw_phi) * self.spins[spin].sx + np.sin(mw_theta) * np.sin(mw_phi) * self.spins[spin].sy + np.cos(mw_theta) * self.spins[spin].sz)
            microwave = self.embed_hamiltonian(microwave, self.spins[spin].order_in_system)
            H += microwave
        return H
    
    def calc_microwave_hamiltonian_RWA(self, B1, omega, phi=0, mw_theta=np.pi/2, mw_phi=0):
        """
        Calculate the microwave Hamiltonian using the rotating wave approximation.

        Parameters:
        - t: The time.
        - B1: The microwave field strength.
        - omega: The microwave frequency.
        - phi: The microwave phase.
        - mw_theta: The polar angle of the microwave field.
        - mw_phi: The azimuthal angle of the microwave field.

        Returns:
        The microwave Hamiltonian using the rotating wave approximation.
        """
        H = 0
        spins = self.spins
        for spin in spins:
            microwave = +omega * self.spins[spin].sz + self.spins[spin].gamma * (B1/2) * (np.cos(phi) * self.spins[spin].sx + np.sin(phi) * self.spins[spin].sy)
            microwave = self.embed_hamiltonian(microwave, self.spins[spin].order_in_system)
            self.microwave = microwave
            H += microwave
        return H
    
    def control_hamiltonian_rwa(self, phi=0):
        """
        Calculate the control Hamiltonian using the rotating wave approximation.

        Parameters:
        - phi: The microwave phase.

        Returns:
        The control Hamiltonian using the rotating wave approximation.
        """
        H = 0
        spins = self.spins
        for spin in spins:
            microwave = self.spins[spin].gamma * (0.5) * (np.cos(phi) * self.spins[spin].sx + np.sin(phi) * self.spins[spin].sy)
            microwave = self.embed_hamiltonian(microwave, self.spins[spin].order_in_system)
            self.microwave = microwave
            H += microwave
        return H
    
    def drift_hamiltonian_rwa(self, omega):
        """
        Calculate the drift Hamiltonian using the rotating wave approximation.

        Parameters:
        - omega: The microwave frequency.

        Returns:
        The control Hamiltonian using the rotating wave approximation.
        """
        H = self.base_hamiltonian_rwa.copy()
        for spin in self.spins:
            detuning = omega * self.spins[spin].sz
            detuning = self.embed_hamiltonian(detuning, self.spins[spin].order_in_system)
            H += detuning
        return H
    
    def control_hamiltonian(self, t, omega):
        """
        Calculate the control Hamiltonian excactly.

        Parameters:
        - t: The time.
        - omega: The microwave frequency.

        Returns:
        The control Hamiltonian.
        """
        return self.microwave_hamiltonian(t, omega)

    def drift_hamiltonian(self):
        """
        Calculate the drift Hamiltonian excactly.

        Returns:
        The control Hamiltonian using the rotating wave approximation.
        """
        return self.base_hamiltonian.copy()
    
    def get_state(self, state_number):
        """
        Get the state of the system.

        Parameters:
        - state_number: The number of the state.

        Returns:
        The state of the system.
        """
        dims = self.H.dims[0]
        return_state = qt.basis(dims, list(np.unravel_index(state_number,dims)))  
        return return_state

    def get_state_dm(self, state_number):
        """
        Get the density matrix of the state.

        Parameters:
        - state_number: The number of the state.

        Returns:
        The density matrix of the state.
        """
        if state_number == -1:
            n = self.H.shape[0]
            probabilities = np.asarray(range(n))/np.sum(range(n))
            return_state_dm = 0
            for i,probability in enumerate(probabilities):
                if probability == 0:
                    continue
                return_state_dm += probability * self.get_state(i).proj()
        else:
            return_state_dm = self.get_state(state_number).proj()
        return return_state_dm
    
    def get_expectation_from_density_matrix(self, density_matrix):
        """
        Get the expectation value for the pure states, from the density matrix.

        Parameters:
        - density_matrix: The density matrix of the state.

        Returns:
        The state of the system.
        """
        dims = self.H.dims[0]
        expect = []
        for i in range(self.H.shape[0]):
            decomposition = list(np.unravel_index(i, dims))
            projector =  qt.basis(dims,decomposition)
            expect.append(qt.expect(density_matrix,projector))
        return np.array(expect)
    
    def get_transition_energies(self):
        """
        Get the transition energies of the system.

        Returns:
        The transition energies with the labels of the Transitions from the simulaters get labels.
        """
        state_labels = Simulator.generate_state_labels(self.H.dims[0],latex=False)
        transition_energies = []
        transition_labels = []
        transitions = {}
        eigenenergies, eigenstates = self.H.eigenstates()
        # sort the eigenstates by looking which entry is the absolute biggest. [-2,5,12] -> 2. Sort the eigenenergies by the eigenstates
        order = np.argsort([np.argmax(np.abs(eigenstate.full())) for eigenstate in eigenstates])
        eigenstates = [eigenstates[i] for i in order]
        eigenenergies = [eigenenergies[i] for i in order]
        # get the transition energies by implementing the selection rules that within the same dimesion the state difference is 1
        # this means for two dimensions -1.5 -0.5 -> -0.5 -0.5 is allowed but -1.5 -0.5 -> -0.5 0.5 is not allowed
        # i need to implement this for any amount of dimensions
        dims = self.H.dims[0]
        for i in range(len(eigenenergies)):
            for j in range(i+1,len(eigenenergies)):
                if np.sum(np.abs(np.array(np.unravel_index(i,dims))-np.array(np.unravel_index(j,dims))))== 1 :
                    transition_energy = np.abs(eigenenergies[j]-eigenenergies[i])
                    transition_label = f"{state_labels[i]} -> {state_labels[j]}"
                    transition_energies.append(transition_energy)
                    transition_labels.append(transition_label)
                    transitions[transition_label] = transition_energy
        self.transitions = transitions
        self.transition_energies = transition_energies
        self.transition_labels = transition_labels
        return transitions, transition_energies, transition_labels
    
    def get_transition_energies_rwa(self):
        """
        Get the transition energies of the system.

        Returns:
        The transition energies with the labels of the Transitions from the simulaters get labels.
        """
        state_labels = Simulator.generate_state_labels(self.H_rwa.dims[0],latex=False)
        transition_energies = []
        transition_labels = []
        transitions = {}
        eigenenergies, eigenstates = self.H_rwa.eigenstates()
        # sort the eigenstates by looking which entry is the absolute biggest. [-2,5,12] -> 2. Sort the eigenenergies by the eigenstates
        order = np.argsort([np.argmax(np.abs(eigenstate.full())) for eigenstate in eigenstates])
        eigenstates = [eigenstates[i] for i in order]
        eigenenergies = [eigenenergies[i] for i in order]
        # get the transition energies by implementing the selection rules that within the same dimesion the state difference is 1
        # this means for two dimensions -1.5 -0.5 -> -0.5 -0.5 is allowed but -1.5 -0.5 -> -0.5 0.5 is not allowed
        # i need to implement this for any amount of dimensions
        dims = self.H_rwa.dims[0]
        for i in range(len(eigenenergies)):
            for j in range(i+1,len(eigenenergies)):
                if np.sum(np.abs(np.array(np.unravel_index(i,dims))-np.array(np.unravel_index(j,dims))))== 1 :
                    transition_energy = np.abs(eigenenergies[j]-eigenenergies[i])
                    transition_label = f"{state_labels[i]} -> {state_labels[j]}"
                    transition_energies.append(transition_energy)
                    transition_labels.append(transition_label)
                    transitions[transition_label] = transition_energy
        self.transitions_rwa = transitions
        self.transition_energies_rwa = transition_energies
        self.transition_labels_rwa = transition_labels
        return transitions, transition_energies, transition_labels



class Simulator():
    def __init__(self, spins=None, D=0, B=0, theta=0, phi=0, H=None, tlist=None, t_start=0, t_end=1, t_steps=10000, psi0=None, psi0_state_number=0):
        """
        Initialize a simulator object.

        Parameters:
        - H: The Hamiltonian of the system.
        - D: The zero field splitting parameter.
        - B: The magnetic field strength.
        - theta: The polar angle of the magnetic field.
        - phi: The azimuthal angle of the magnetic field.
        - spins: A dict of Spin objects.
        - tlist: The time list for the evolution.
        - t_start: The start time of the evolution.
        - t_end: The end time of the evolution.
        - psi0: The initial state of the system.
        - psi0_state_number: The state number of the initial state.

        Returns:
        None
        """
        if H is not None:
            self.hamiltonian = H
        else:
            self.hamiltonian = Hamiltonian(D, B, theta, phi, spins)
        if tlist is not None:
            self.tlist = tlist
        else:
            self.tlist = np.linspace(t_start, t_end, t_steps)
        if psi0 is not None:
            self.psi0 = psi0
        else:
            self.psi0 = self.hamiltonian.get_state_dm(psi0_state_number)

    def simulate(self, psi0=None, psi0_state_number=None, H=None, tlist=None, t_start=0, t_end=1, t_steps=1000, 
                 B1=0, omega=0, phi=0, c_ops=[], RWA=False, progress_bar="enhanced", nsteps=10000):
        """
        Simulate the system.

        Parameters:
        - psi0: The initial state of the system.
        - psi0_state_number: The state number of the initial state.
        - H: The Hamiltonian of the system.
        - tlist: The time list for the evolution.
        - t_start: The start time of the evolution.
        - t_end: The end time of the evolution.
        - B1: The microwave field strength.
        - omega: The microwave frequency.
        - phi: The microwave phase.
        - c_ops: The collapse operators.
        - RWA: Whether to use the rotating wave approximation. This works only for electron transitions.
        - progress_bar: The type of progress bar to use.
        - nsteps: The number of steps in the simulation.
        
        Returns:
        The result of the simulation.
        """
        self.B1 = B1
        self.omega = omega
        self.phi = phi
        self.c_ops = c_ops
        if psi0 is not None:
            self.psi0 = psi0
        if psi0_state_number is not None:
            self.psi0 = self.hamiltonian.get_state_dm(psi0_state_number)
        if H is not None:
            self.hamiltonian = H
        if tlist is not None:
            self.tlist = tlist
        if t_start != 0 or t_end != 1:
            self.tlist = np.linspace(t_start, t_end, t_steps)
        options_dict = {"nsteps": nsteps, "progress_bar": progress_bar}
        if B1 != 0 and omega != 0:
            if RWA:
                if isinstance(B1, list):
                    if progress_bar == "enhanced":
                        sim_start_time = time.time()
                    B1_amplitudes = B1[0]
                    B1_timegrid = B1[1]
                    options_dict["progress_bar"] = ""
                    # for every time intervall in timegrid get the amplitude and run a simulation 
                    # with the amplitude and the final state from the last simulation
                    # append all the states into the result
                    result = [self.psi0]
                    previous_end_time = 0
                    for i in range(len(B1_timegrid)-1):
                        # Define the current interval
                        start_time = B1_timegrid[i]
                        end_time = B1_timegrid[i+1]
                        # Adjust the end_time to not exceed the duration
                        if end_time > t_end:
                            end_time = t_end
                        if start_time >= t_end:
                            break
                        # take all the elements from the tlist that are in the intervall of the timegrid and make a sub tlist
                        tsublist = list(self.tlist[np.logical_and(self.tlist >= start_time, self.tlist < end_time)])
                        if not tsublist:
                            break
                        tsublist.insert(0, previous_end_time)
                        res = qt.mesolve(self.hamiltonian.calc_hamiltonian(1, B1_amplitudes[i], omega, phi, RWA=True)*2*np.pi, result[-1], tsublist, self.c_ops, [], options=options_dict).states
                        result.extend(res[1:])
                        previous_end_time = tsublist[-1]
                    if progress_bar == "enhanced":
                        print(f"Total run time:   {time.time()-sim_start_time}s")
                else:
                    result = qt.mesolve(self.hamiltonian.calc_hamiltonian(1, B1, omega, phi, RWA=True)*2*np.pi, self.psi0, self.tlist, self.c_ops, [], options=options_dict).states
            else:
                result = qt.mesolve(lambda t: self.hamiltonian.calc_hamiltonian(t, B1, omega, phi)*2*np.pi, self.psi0, self.tlist, self.c_ops, [], options=options_dict).states
        else:
            result = qt.mesolve(self.hamiltonian.H*2*np.pi, self.psi0, self.tlist, self.c_ops, [], options=options_dict).states
        return result
    
    def analyse(self, result, states=None, sim_mw=False, legend=True, multiple=False): 
        """
        Analyse the result of the simulation.

        Parameters:
        - result: The result of the simulation.
        - states: The states to analyse.
        - sim_mw: Whether to simulate the microwave field.
        - RWA: Whether to use the rotating wave approximation.

        Returns:
        None
        """
        plt.rcParams['text.usetex'] = True  # to use LaTeX in figures
        plt.rcParams["text.latex.preamble"]= r'\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage{lmodern, mathpazo}\inputencoding{utf8}\usepackage{amsmath}\usepackage{amssymb}\usepackage{dsfont}\usepackage{mathtools}\usepackage{physics}\usepackage{siunitx}'
        plt.rcParams['font.family'] = ['serif']
        #plt.figure(figsize=(10,5))
        # Plot the expactation values of a all the pure states
        if not multiple:
            fig = plt.figure(figsize=(11, 7))
            ax = fig.add_subplot(111)
        else:
            ax = plt.gca()
        legend_label = r"{states}"
        state_labels = Simulator.generate_state_labels(self.hamiltonian.H.dims[0])
        expectation_values = []
        states_to_plot = range(self.hamiltonian.H.shape[0]) if states is None else states
        for i in states_to_plot:
            expectation_values.append(qt.expect(self.hamiltonian.get_state_dm(len(states_to_plot)-i-1), result))
            plt.plot(self.tlist, expectation_values[-1], label=legend_label.format(states=state_labels[len(states_to_plot)-i-1]))
        if sim_mw:
            plt.plot(self.tlist, self.B1 * np.cos(self.omega * self.tlist * 2 * np.pi + self.phi), label="Microwave field")
        plt.xlabel(r'Time $t$ [\SI{}{\micro\second}]', fontsize=28)
        plt.ylabel(r'Expectation values', fontsize=28)
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.grid()
        if legend: 
            plt.legend(loc='upper right', fontsize=14)
        #plt.savefig("SlideMW.png", format="png", bbox_inches="tight", dpi=300)
        #plt.show()
    
    def generate_state_labels(dims, latex=True):
        """
        Generate the state labels.

        Parameters:
        - dims: The dimensions of the system.

        Returns:
        The state labels.
        """
        dim_states = []
        for dim in dims:
            # Generate a list of states from -(dim - 1)*0.5 to (dim - 1)*0.5 in steps of 0.5
            states = [state - (dim - 1)*0.5 for state in range(dim)]
            state_strings = []
            for state in states:
                if state.is_integer():
                    state_strings.append(str(int(state)))
                else:
                    numerator = int(state * 2)
                    sign = "+" if numerator >= 0 else "-"
                    if latex:
                        state_strings.append(sign + r"\frac{" + str(np.abs(numerator)) + "}{2}")
                    else:
                        state_strings.append(f"{numerator}/2")
            dim_states.append(state_strings)
        combinations = list(itertools.product(*dim_states))
        if latex:
            #state_labels = [r"$\ket{" + r"}_{^\mathrm{e}}\otimes\ket{".join(combination) + r"}_{^{29}\mathrm{Si}}$" for combination in combinations]
            state_labels = [r"$\ket{" + r", ".join(combination) + "}$" for combination in combinations]
        else:
            state_labels = [r"|" + r" ".join(combination) + r">" for combination in combinations]
        return state_labels
    
    def get_expectation(self, result):
        """
        Calculate the expectation value of the result for each pure state.

        Parameters:
        - result: The result of the simulation.

        Returns:
        The expectation value of each state.
        """
        return np.asarray([self.hamiltonian.get_expectation_from_density_matrix(state) for state in result])
    
class SiC(AbstractFoM):
    def __init__(self, args_dict: dict = None, optimization=None, func_eval_amount=1000, RWA=False, dur=0.3, initial_guess=None,
                 optimize_det=True, optimize_rabi_error=True, optimize_mw_noise=False, optimize_mw_length=False):
        if args_dict is None:
            args_dict = {}

        self.FoM_list = []
        self.param_list = []
        self.save_path = ""
        self.func_eval_num = 0
        self.start_time_func_eval = time.time()
        self.func_eval_amount = func_eval_amount
        self.RWA = RWA
        self.optimization = optimization
        self.dur = dur
        self.initial_guess = initial_guess
        self.optimize_det = optimize_det
        self.optimize_rabi_error = optimize_rabi_error
        self.optimize_mw_noise = optimize_mw_noise
        self.optimize_mw_length = optimize_mw_length
        self.mw_averages = args_dict.get("mw_averages", 4)

    def save_FoM(self):
        np.savetxt(os.path.join(self.save_path, 'FoM.txt'), self.FoM_list)
        np.savetxt(os.path.join(self.save_path, 'params.txt'), self.param_list)

    def set_save_path(self, save_path: str = ""):
        self.save_path = save_path

    def get_control_Hamiltonians(self):
        return self.optimization.H.control_hamiltonian_rwa().full()

    def get_drift_Hamiltonian(self):
        return self.optimization.H.drift_hamiltonian_rwa(self.optimization.H.transitions['|-3/2 -1/2> -> |-1/2 -1/2>']).full()

    def get_target_state(self):
        return self.optimization.U_t.full()

    def get_initial_state(self):
        return self.optimization.U_0.full()

    def get_propagator(self, pulses_list: list = [], time_grids_list: list = [], parameters_list: list = []) -> np.array:
        B1_pulse = [pulses_list[0].tolist(),time_grids_list[0].tolist()]
        res = self.optimization.simulator.simulate(B1=B1_pulse, omega=self.optimization.H.transitions['|-3/2 -1/2> -> |-1/2 -1/2>'], phi=0, RWA=self.RWA, progress_bar="", t_end=self.dur, nsteps=self.optimization.nsteps)
        return [result.full() for result in res]
    
    def get_FoM(self, pulses: list = [], parameters: list = [], timegrids: list = []) -> dict:
        """
        Compute and return the figure of merit
        :param pulses: List of pulses
        :param parameters: List of parameters
        :param timegrids: List of time grids
        :return dict: Figure of merit in a dictionary
        """
        if self.initial_guess is not None:
            B1_pulse=self.initial_guess(*parameters)
        else:
            B1_pulse = [pulses[0].tolist(),timegrids[0].tolist()]
            #dur = parameters[0] only if dur gets optimized as parameter
        dur = self.dur
        self.func_eval_num += 1
        if self.func_eval_num % 100 == 0:
            print(f"{self.func_eval_num}/{self.func_eval_amount}:{time.time()-self.start_time_func_eval} s. Remaining time: {(time.time()-self.start_time_func_eval)*(self.func_eval_amount-self.func_eval_num)/self.func_eval_num} s")

        fidelity = self.optimization.fidelity_func(B1_pulse, dur, RWA=self.RWA, optimize_det=self.optimize_det, optimize_rabi_error=self.optimize_rabi_error, optimize_mw_noise=self.optimize_mw_noise, mw_averages=self.mw_averages, optimize_mw_length=self.optimize_mw_length)
        self.FoM_list.append(fidelity)
        self.param_list.append(parameters)
        self.pulse=pulses

        return {"FoM": fidelity}

    
class Optimization():
    def __init__(self, H: Hamiltonian, direction="minimization"):
        """
        Initialize an optimizer object.

        Parameters:
        - H: The Hamiltonian of the system.
        - direction: The direction of the optimization. (default: "minimization")

        Returns:
        None
        """
        self.H = H
        self.direction = direction
        # self.H0 = self.H.base_hamiltonian
        # self.Hc = self.H.calc_microwave_hamiltonian_RWA(1,H1.transitions['|-3/2 -1/2> -> |-1/2 -1/2>'])
        self.U_0 = self.H.get_state_dm(-1)
        self.G_t = self.cnot([2,1,0,3,4,5,6,7])
        self.U_t = self.G_t*self.U_0*self.G_t.dag()
        self.simulator = Simulator(H=self.H, t_start=0, t_end=0.5, t_steps=1000, psi0=self.U_0)
        folder = os.getcwd()
        self.optimization_dictionary = readjson(os.path.join(folder, "opt_dictionary1.json"))
        self.nsteps = 10000

    
    def fidelity_func(self, B1_pulse, dur, RWA=False, optimize_det=True, optimize_rabi_error=True,
                      optimize_mw_noise=False, mw_averages=4, optimize_mw_length=False) -> float:
        """
        Calculate the FoM of the system.

        Parameters:
        - B1_pulse: The microwave pulse.
        - dur: The duration of the pulse.
        - RWA: Whether to use the rotating wave approximation.
        - optimize_det: Whether to optimize the detuning.
        - optimize_rabi_error: Whether to optimize the Rabi error.
        - optimize_mw_noise: Whether to optimize the microwave noise.
        - mw_averages: The number of averages for the microwave noise.
        - optimize_mw_length: Whether to optimize the microwave length.
        
        Returns:
        The FoM of the system.
        """
        if RWA:
            transition = self.H.transitions_rwa['|-3/2 -1/2> -> |-1/2 -1/2>']
        else:
            transition = self.H.transitions['|-3/2 -1/2> -> |-1/2 -1/2>']        
        res = self.simulator.simulate(B1=B1_pulse, omega=transition, phi=0, RWA=RWA, progress_bar="", t_end=dur, nsteps=self.nsteps)
        U_f = res[-1]

        center_weight = 2
        fidelity = center_weight*self.state_fidelity(U_f, self.U_t)
        weights = center_weight
        if optimize_det:
            detunings=[0.5,1.0,-0.5,-1.0,1.75,-1.75]
            detuning_weights=[0.95,0.80,0.95,0.80,0.5,0.5]
            weights += np.sum(detuning_weights)
            for i,det in enumerate(detunings):
                res = self.simulator.simulate(B1=B1_pulse, omega=transition+det, phi=0, RWA=RWA, progress_bar="", t_end=dur, nsteps=self.nsteps)
                U_f = res[-1]
                f_i = self.state_fidelity(U_f, self.U_t)
                fidelity+=detuning_weights[i]*f_i

        if optimize_rabi_error:
            #rabiErrors = [0.05,-0.05,0.1,-0.1, 0.15,-0.15]
            #rabiErrors = [0.1,-0.1,0.20,-0.20, 0.3,-0.3]
            rabiErrors = [0.1,-0.1,0.15,-0.15,0.20,-0.20,0.25,-0.25, 0.3,-0.3]
            rabi_weights = [0.95,0.95,0.875,0.875,0.8,0.8,0.65,0.65,0.5,0.5]
            weights += 0.5*np.sum(rabi_weights)
            B1_pulse_amplitudes = np.array(B1_pulse[0])
            B1_pulse_times = B1_pulse[1]
            for i,rE in enumerate(rabiErrors):
                B1_pulse_rabi_amplitudes = B1_pulse_amplitudes * (1 + rE)
                B1_pulse_rabi = [B1_pulse_rabi_amplitudes.tolist(), B1_pulse_times]
                res = self.simulator.simulate(B1=B1_pulse_rabi, omega=transition, phi=0, RWA=RWA, progress_bar="", t_end=dur, nsteps=self.nsteps)
                U_f = res[-1]
                f_i = self.state_fidelity(U_f, self.U_t)
                fidelity+=0.5*rabi_weights[i]*f_i

        if optimize_mw_length:
            length_errors = [0.1, -0.1, 0.25, -0.25, 0.4, -0.4]
            length_weights = [0.95, 0.95, 0.8, 0.8, 0.5, 0.5]
            weights += np.sum(length_weights)
            B1_pulse_amplitudes = B1_pulse[0]  # Keep amplitudes unchanged
            B1_pulse_times = np.array(B1_pulse[1])
            for i,lE in enumerate(length_errors):
                B1_pulse_length_adjusted_times = B1_pulse_times * (1 + lE)
                B1_pulse_length_adjusted = [B1_pulse_amplitudes, B1_pulse_length_adjusted_times.tolist()]
                res = self.simulator.simulate(B1=B1_pulse_length_adjusted, omega=transition, phi=0, RWA=RWA, progress_bar="", t_end=dur*(1 + lE), nsteps=self.nsteps)
                U_f = res[-1]
                f_i = self.state_fidelity(U_f, self.U_t)
                fidelity+=length_weights[i]*f_i

        if optimize_mw_noise:
            microwave_noise = [0.05,0.15,0.25]
            microwave_noise_weights = [0.95,0.9,0.8]
            weights += np.sum(microwave_noise_weights)
            B1_pulse_amplitudes = np.array(B1_pulse[0])
            B1_pulse_times = B1_pulse[1]
            number_of_averages = mw_averages
            # add random noise to the microwave pulse
            for i,mn in enumerate(microwave_noise):
                f_i = 0
                for j in range(number_of_averages):
                    B1_pulse_noise = [(B1_pulse_amplitudes+np.random.normal(0,mn,len(B1_pulse_amplitudes))).tolist(), B1_pulse_times]
                    res = self.simulator.simulate(B1=B1_pulse_noise, omega=transition, phi=0, RWA=RWA, progress_bar="", t_end=dur, nsteps=self.nsteps)
                    U_f = res[-1]
                    f_i += self.state_fidelity(U_f, self.U_t)
                fidelity+=3*microwave_noise_weights[i]*(f_i/number_of_averages)
        
        
        fidelity /= weights
        #fidelity += dur*0.00001
        
        return fidelity
    
    def state_fidelity(self, phi_f: qt.Qobj, phi_t: qt.Qobj, compare_target_states=False) -> float:
        """
        Calculate the state fidelity of the system.

        Parameters:
        - phi_f: The final density matrix.
        - phi_t: The target density matrix.
        - compare_target_states: Whether to compare the target states.

        Returns:
        The state fidelity of the system.
        """
        final_expect = self.H.get_expectation_from_density_matrix(phi_f)
        target_expect = self.H.get_expectation_from_density_matrix(phi_t)
        beginning_expect = self.H.get_expectation_from_density_matrix(self.U_0)
        dim = target_expect.shape[0]
        if compare_target_states:
            state_weights = np.zeros(dim)
            state_weights[0] = 2
            state_weights[2] = 2
        else:
            state_weights = np.ones(dim)*0.5
        f = np.sum(state_weights*((target_expect-final_expect)**2))/np.sum(((target_expect-beginning_expect)**2))
        return f
    
    def cnot(self, order, H=None) -> qt.Qobj:
        """
        Generate a CNOT gate.

        Parameters:
        - H: The Hamiltonian of the system.
        - order: The order of the states after the gate.

        Returns:
        The CNOT gate.
        """
        cnot_gate = 0
        if H is None:
            H = self.H
        for i,j in enumerate(order):
            cnot_gate += H.get_state(i)*H.get_state(j).dag()
        return cnot_gate
    
    def square_pulse(self, amplitude, number_bins, dur):
        """
        Generate a square pulse.

        Parameters:
        - amplitude: The amplitude of the pulse.
        - number_bins: The number of bins in the pulse.
        - dur: The duration of the pulse.

        Returns:
        A list containing the pulse amplitude and time grid.
        """
        return [list(np.ones(number_bins) * amplitude), list(np.linspace(0, dur, number_bins))]
    
    def sinc_pulse(self, amplitude, a, b, frequency, number_bins, dur):
        """
        Generate a sinc pulse.

        Parameters:
        - amplitude: The amplitude of the pulse.
        - a: Parameter a.
        - b: Parameter b.
        - frequency: The frequency of the pulse.
        - number_bins: The number of bins in the pulse.
        - dur: The duration of the pulse.

        Returns:
        A list containing the pulse amplitude and time grid.
        """
        timegrid = np.linspace(0, dur, number_bins)
        sinc_values = np.sinc(frequency * (timegrid - a))
        
        # Scale the sinc values by the amplitude
        amplitudes = amplitude * sinc_values + b
        # Cut off amplitudes at ±2
        amplitudes = np.clip(amplitudes, -2, 2)
        return [list(amplitudes), list(timegrid)]
    
    def hermite_envelope_pulse(self, amplitude, a, b, c, number_bins, dur):
        """
        Generate a pulse with a Hermite envelope.

        Parameters:
        - amplitude: The amplitude of the pulse.
        - a: Parameter a.
        - b: Parameter b.
        - c: Parameter c.
        - number_bins: The number of bins in the pulse.
        - dur: The duration of the pulse.

        Returns:
        A list containing the pulse amplitude and time grid.
        """
        ## In 2403.10633v1 b=dur/2 c=0.956/((0.1667*dur)**2) and a=1/((0.1667*dur)**2)
        timegrid = np.linspace(0, dur, number_bins)
        envelope = (1 - c * (timegrid - b)**2)*np.exp(-a * (timegrid - b)**2) 
        amplitudes = amplitude * envelope
        amplitudes = np.clip(amplitudes, -2, 2)
        return [list(amplitudes), list(timegrid)]
    
    def optimize(self, max_eval_total=1000, RWA=False, cbs_funct_evals= 500, 
                 basis_vector_number=10, basis_upper_limit=2.0, dur=0.3, nsteps=10000, bins_number=200, 
                 initial_guess=None, optimize_det=True, optimize_rabi_error=True, optimize_mw_noise=False, mw_averages=4,
                 optimize_parameters=True, grape=False, optimize_mw_length=False):
        """
        Optimize the pulse.

        Parameters:
        - max_eval_total: The maximum number of evaluations.
        - RWA: Whether to use the rotating wave approximation.
        - cbs_funct_evals: The number of evaluations for the change based stopping criteria.
        - basis_vector_number: The number of basis vectors.
        - basis_upper_limit: The upper limit of the basis frequencies in MHz.
        - dur: The duration of the pulse.
        - nsteps: The number of steps in the simulation.
        - bins_number: The number of bins in the pulse.
        - initial_guess: The initial guess of the pulse.
        - optimize_det: Whether to optimize the detuning.
        - optimize_rabi_error: Whether to optimize the Rabi error.
        - optimize_mw_noise: Whether to optimize the microwave noise.
        - mw_averages: The number of averages for the microwave noise.
        - optimize_mw_length: Whether to optimize the microwave pulse length.
        - optimize_parameters: Whether to optimize the parameters.
        - grape: Whether to use GRAPE optimization.

        Returns:
        The resulting pulse of the optimization.
        """
        pulses_list = []
        self.dur = dur
        self.nsteps = nsteps
        self.mw_averages = mw_averages
        args_dict = {"mw_averages": mw_averages}
        self.FoM_object = SiC(args_dict=args_dict, optimization=self, func_eval_amount=max_eval_total, RWA=RWA, dur=dur, optimize_det=optimize_det, optimize_rabi_error=optimize_rabi_error, optimize_mw_noise=optimize_mw_noise, optimize_mw_length=optimize_mw_length)
        self.optimization_dictionary["algorithm_settings"]["max_eval_total"]= max_eval_total
        self.optimization_dictionary["algorithm_settings"]["dsm_settings"]["stopping_criteria"]["change_based_stop"]["cbs_funct_evals"]= cbs_funct_evals
        
        if not optimize_parameters:
            self.optimization_dictionary["parameters"] = []

        self.optimization_dictionary["pulses"][0]["bins_number"]= bins_number
        if initial_guess is None:
            initial_guess = np.ones(bins_number)*0.998*(0.2/dur)
        self.optimization_dictionary["pulses"][0]["initial_guess"]["list_function"]=initial_guess
        self.optimization_dictionary["pulses"][0]["basis"]["basis_vector_number"]= basis_vector_number
        self.optimization_dictionary["pulses"][0]["basis"]["random_super_parameter_distribution"]["upper_limit"]= basis_upper_limit
        
        self.optimization_dictionary["times"][0]["initial_value"]= dur

        if grape:
            self.optimization_dictionary["optimization_client_name"] = "SiC_GRAPE"
            self.optimization_dictionary["algorithm_settings"]["algorithm_name"] = "GRAPE"
            del self.optimization_dictionary['algorithm_settings']['dsm_settings']
            del self.optimization_dictionary["pulses"][0]["initial_guess"]
            self.optimization_dictionary["pulses"][0]["amplitude_variation"] = 0.3
            #self.optimization_dictionary["pulses"][0]["basis"]  = {"basis_name": "PiecewiseBasis"}

        optimization_obj = Optimizer(self.optimization_dictionary, self.FoM_object)
        self.results_path = optimization_obj.results_path
        self.FoM_object.set_save_path(self.results_path)
        start_time = time.time()
        optimization_obj.execute()
        end_time = time.time()
        print(f"Execution time: {end_time-start_time} seconds")
        self.FoM_object.save_FoM()

        self.opt_alg_obj = optimization_obj.get_optimization_algorithm()
        self.opt_controls =self.opt_alg_obj.get_best_controls()
        self.fomlist = self.opt_alg_obj.FoM_list

        pulse, timegrid = self.opt_controls["pulses"][0], self.opt_controls["timegrids"][0]
        
        np.savetxt(os.path.join(self.results_path, 'bestControls.txt'), [pulse, timegrid])
        if optimize_parameters:
            dur = self.opt_controls["parameters"][0]
            np.savetxt(os.path.join(self.results_path, 'bestParams.txt'), [dur])
            return pulse, timegrid, dur
        else: 
            return pulse, timegrid
    
    def optimize_initial_guess(self, max_eval_total=1000, RWA=False, dur=0.3, nsteps=10000, initial_guess_function=None, initial_guess_parameters=None,
                               optimize_det=True, optimize_rabi_error=True, optimize_mw_noise=False, mw_averages=4, optimize_mw_length=False):
        """
        Optimize the initial guess for the pulse.

        Parameters:
        - max_eval_total: The maximum number of evaluations.
        - RWA: Whether to use the rotating wave approximation.
        - dur: The duration of the pulse.
        - nsteps: The number of steps in the simulation.
        - optimize_det: Whether to optimize the detuning.
        - optimize_rabi_error: Whether to optimize the Rabi error.
        - optimize_mw_noise: Whether to optimize the microwave noise.
        - mw_averages: The number of averages for the microwave noise.
        - optimize_mw_length: Whether to optimize the microwave pulse length.
        - initial_guess_function: The initial guess function.
        - initial_guess_parameters: The initial guess parameters.

        Returns:
        The resulting pulse of the optimization.
        """
        self.mw_averages = mw_averages
        pulses_list = []
        self.dur = dur
        self.nsteps = nsteps
        self.FoM_object = SiC(optimization=self, func_eval_amount=max_eval_total, RWA=RWA, dur=dur, 
                              initial_guess=initial_guess_function, optimize_det=optimize_det, 
                              optimize_rabi_error=optimize_rabi_error, optimize_mw_noise=optimize_mw_noise, optimize_mw_length=optimize_mw_length)
        optimization_dictionary = {"optimization_client_name": "ParametersDirectSearch"}
        optimization_dictionary["algorithm_settings"] = {"algorithm_name": "DirectSearch"}
        dsm_settings = {
                "general_settings": {
                    "dsm_algorithm_name": "NelderMead",
                    "is_adaptive": False
                },
                "stopping_criteria": {
                    "xatol": 1e-5,
                    "fatol": 1e-12
                }
            }
        optimization_dictionary["algorithm_settings"]["dsm_settings"] = dsm_settings
        optimization_dictionary["algorithm_settings"]["max_eval_total"] = max_eval_total

        optimization_dictionary["pulses"] = []
        optimization_dictionary["times"] = []


        total_number_of_parameters = len(initial_guess_parameters)
        parameters = []
        for index in range(total_number_of_parameters):
            parameters.append({"parameter_name": "Parameter{0}".format(index),
                            "lower_limit": -2.0,
                            "upper_limit": 2.0,
                            "initial_value": initial_guess_parameters[index],
                            "amplitude_variation": 0.1})
        optimization_dictionary["parameters"] = parameters

        optimization_obj = Optimizer(optimization_dictionary, self.FoM_object)
        self.results_path = optimization_obj.results_path
        self.FoM_object.set_save_path(self.results_path)
        start_time = time.time()
        optimization_obj.execute()
        end_time = time.time()
        print(f"Execution time: {end_time-start_time} seconds")
        self.FoM_object.save_FoM()

        self.opt_alg_obj = optimization_obj.get_optimization_algorithm()
        self.opt_controls =self.opt_alg_obj.get_best_controls()
        self.fomlist = self.opt_alg_obj.FoM_list

        # it contains the pulses and time grids under certain keys as a dictionary
        params = self.opt_controls["parameters"]
        np.savetxt(os.path.join(self.results_path, 'bestParams.txt'), params)

        print("Plotting the pulse")

        # it contains the pulses and time grids under certain keys as a dictionary
        p = initial_guess_function(*params)
        pulse, timegrid = p[0], p[1]

        # Plot the pulse over time
        fig = plt.figure(figsize=(11, 7))
        ax = fig.add_subplot(111)
        plt.subplots_adjust(bottom=0.15, top=0.9, right=0.98, left=0.1)
        plt.step(timegrid, pulse, linewidth=1.5, zorder=10, label=r"Initial Pulse")
        plt.grid(True, which="both")
        plt.xlabel(r'Time $t$ [\SI{}{\micro\second}]', fontsize=14)
        plt.ylabel(r'Amplitude $B$ [\SI{}{\gauss}]', fontsize=14)
        plt.legend()
        plt.savefig(os.path.join(self.results_path,"Pulse.pdf"), format="pdf", bbox_inches="tight")
        
        return params

    def analyse_optimization(self, initial_guess_function=False, initial_guess_params=None):
        """
        Analyse the optimization.

        Parameters:

        Returns:
        None
        """
        plt.rcParams['text.usetex'] = True  # to use LaTeX in figures
        plt.rcParams["text.latex.preamble"]= r'\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage{lmodern, mathpazo}\inputencoding{utf8}\usepackage{amsmath}\usepackage{amssymb}\usepackage{dsfont}\usepackage{mathtools}\usepackage{physics}\usepackage{siunitx}\DeclareSIUnit\gauss{G}'
        plt.rcParams['font.family'] = ['serif']
        
        print("Plotting the FoM")

        fig = plt.figure(figsize=(11, 7))
        ax = fig.add_subplot(111)
        iterations = range(1, len(self.fomlist)+1)
        ax.plot(iterations, np.asarray(self.fomlist), linewidth=1.5, zorder=10, label=r"Fidelity")
        ax.scatter(iterations, np.asarray(self.fomlist), s=15)
        plt.grid(True, which="both")
        plt.xlabel(r'Function Evaluation', fontsize=14)
        plt.ylabel(r'FoM [a.u.]', fontsize=14)
        plt.yscale("log")
        plt.legend()
        plt.savefig(os.path.join(self.results_path,"FoM.pdf"), format="pdf", bbox_inches="tight")


        print("Plotting the pulse")

        if initial_guess_function:
            p = initial_guess_function(*initial_guess_params)
            pulse, timegrid = p[0], p[1]
        else:
            pulse, timegrid, dur = self.opt_controls["pulses"][0], self.opt_controls["timegrids"][0], self.opt_controls["parameters"][0]
        dur = self.dur
        dur_index = np.searchsorted(timegrid, dur, side='right') - 1

        # Plot the pulse over time
        fig = plt.figure(figsize=(11, 7))
        ax = fig.add_subplot(111)
        plt.subplots_adjust(bottom=0.15, top=0.9, right=0.98, left=0.1)
        plt.step(timegrid[:dur_index], pulse[:dur_index], linewidth=1.5, zorder=10, label=r"Pulse")
        plt.grid(True, which="both")
        plt.xlabel(r'Time $t$ [\SI{}{\micro\second}]', fontsize=14)
        plt.ylabel(r'Amplitude $B$ [\SI{}{\gauss}]', fontsize=14)
        plt.legend()
        plt.savefig(os.path.join(self.results_path,"Pulse.pdf"), format="pdf", bbox_inches="tight")

        print("Simulating the pulse")

        B1_pulse = [pulse,timegrid]
        result = self.simulator.simulate(B1=B1_pulse, omega=self.H.transitions['|-3/2 -1/2> -> |-1/2 -1/2>'], phi=0, RWA=False, progress_bar="enhanced", t_end=dur)
        fig = plt.figure(figsize=(11, 7))
        ax = fig.add_subplot(111)
        plt.subplots_adjust(bottom=0.15, top=0.9, right=0.98, left=0.1)
        legend_label = r"{states}"
        state_labels = Simulator.generate_state_labels(self.H.H.dims[0])
        expectation_values = []
        states_to_plot = range(self.H.H.shape[0])
        for i in states_to_plot:
            expectation_values.append(qt.expect(self.H.get_state_dm(i), result))
            plt.plot(self.simulator.tlist, expectation_values[-1], label=legend_label.format(states=state_labels[i]))
        plt.xlabel(r'Time $t$ [\SI{}{\micro\second}]', fontsize=14)
        plt.ylabel(r'Expectation values', fontsize=14)
        plt.grid(True, which="both")
        plt.legend()
        plt.savefig(os.path.join(self.results_path,"Simulation.pdf"), format="pdf", bbox_inches="tight")
    #   time_array = np.array(self.simulator.tlist)
    #   expectation_array = np.array(expectation_values)    
    #   np.savetxt(os.path.join(self.results_path, 'Simulation_data.txt'), [self.simulator.tlist,expectation_values])


        # simulate the pulse for detuning between -5 and 5 and plot the fidelities
        print("Analyzing the detuning stability")

        detunings = np.linspace(-15,15,100)
        fidelities = []
        for det in detunings:
            res = self.simulator.simulate(B1=B1_pulse, omega=self.H.transitions['|-3/2 -1/2> -> |-1/2 -1/2>']+det, phi=0, RWA=False, progress_bar="", t_end=dur)
            U_f = res[-1]
            fidelities.append(1-self.state_fidelity(U_f,self.U_t))
        fig = plt.figure(figsize=(11, 7))
        ax = fig.add_subplot(111)
        plt.subplots_adjust(bottom=0.15, top=0.9, right=0.98, left=0.1)
        plt.plot(detunings, fidelities, linewidth=1.5, zorder=10, label=r"Detuning Fidelity")
        plt.grid(True, which="both")
        plt.xlabel(r'Detuning $\Delta\omega$ [$\SI{}{\mega\hertz}$]', fontsize=14)
        plt.ylabel(r'FoM [a.u.]', fontsize=14)
        plt.legend()
        plt.savefig(os.path.join(self.results_path,"DetuningStability.pdf"), format="pdf", bbox_inches="tight")
        np.savetxt(os.path.join(self.results_path, 'DetuningStability_data.txt'), [detunings,fidelities])
        
        print("Analyzing the Rabi Error stability")

        rabiErrors = np.linspace(-0.5, 0.5, 100)
        fidelities = []
        for i, rE in enumerate(rabiErrors):
            B1_pulse_amplitudes = np.array(B1_pulse[0])
            B1_pulse_times = B1_pulse[1]
            B1_pulse_rabi_amplitudes = B1_pulse_amplitudes * (1 + rE)
            B1_pulse_rabi = [B1_pulse_rabi_amplitudes.tolist(), B1_pulse_times]
            res = self.simulator.simulate(B1=B1_pulse_rabi, omega=self.H.transitions['|-3/2 -1/2> -> |-1/2 -1/2>'], phi=0, RWA=False, progress_bar="", t_end=dur)
            U_f = res[-1]
            fidelities.append(1-self.state_fidelity(U_f, self.U_t)) 
        fig = plt.figure(figsize=(11, 7))
        ax = fig.add_subplot(111)
        plt.subplots_adjust(bottom=0.15, top=0.9, right=0.98, left=0.1)
        plt.plot(rabiErrors, fidelities, linewidth=1.5, zorder=10, label=r"Rabi Error Fidelity")
        plt.grid(True, which="both")
        plt.xlabel(r'Rabi Error', fontsize=14)
        plt.ylabel(r'FoM [a.u.]', fontsize=14)
        plt.legend()
        plt.savefig(os.path.join(self.results_path,"RabiStability.pdf"), format="pdf", bbox_inches="tight")
        np.savetxt(os.path.join(self.results_path, 'RabiStability_data.txt'), [rabiErrors,fidelities])

        print("Analyzing the Microwave Pulse Length Error stability")

        length_errors = np.linspace(-0.5, 0.5, 100)
        fidelities = []
        for i, lE in enumerate(length_errors):
            B1_pulse_amplitudes = B1_pulse[0]
            B1_pulse_times = np.array(B1_pulse[1])
            B1_pulse_length_adjusted_times = B1_pulse_times * (1 + lE)
            B1_pulse_length_adjusted = [B1_pulse_amplitudes, B1_pulse_length_adjusted_times.tolist()]
            res = self.simulator.simulate(B1=B1_pulse_length_adjusted, omega=self.H.transitions['|-3/2 -1/2> -> |-1/2 -1/2>'], phi=0, RWA=False, progress_bar="", t_end=dur*(1 + lE))
            U_f = res[-1]
            fidelities.append(1-self.state_fidelity(U_f, self.U_t)) 
        fig = plt.figure(figsize=(11, 7))
        ax = fig.add_subplot(111)
        plt.subplots_adjust(bottom=0.15, top=0.9, right=0.98, left=0.1)
        plt.plot(length_errors, fidelities, linewidth=1.5, zorder=10, label=r"Length Error Fidelity")
        plt.grid(True, which="both")
        plt.xlabel(r'Length Error', fontsize=14)
        plt.ylabel(r'FoM [a.u.]', fontsize=14)
        plt.legend()
        plt.savefig(os.path.join(self.results_path,"LengthStability.pdf"), format="pdf", bbox_inches="tight")
        np.savetxt(os.path.join(self.results_path, 'LengthStability_data.txt'), [length_errors,fidelities])

        print("Analyzing the Microwave Noise stability")

        microwave_noise_levels = np.linspace(0, 0.5, 100)
        fidelities = []
        number_of_averages = self.mw_averages
        for i, mn in enumerate(microwave_noise_levels):
            f_i = 0
            B1_pulse_amplitudes = np.array(B1_pulse[0])
            B1_pulse_times = B1_pulse[1]
            for j in range(number_of_averages):
                B1_pulse_noise = [(B1_pulse_amplitudes + np.random.normal(0, mn, len(B1_pulse_amplitudes))).tolist(), B1_pulse_times]
                res = self.simulator.simulate(B1=B1_pulse_noise, omega=self.H.transitions['|-3/2 -1/2> -> |-1/2 -1/2>'], phi=0, RWA=False, progress_bar="", t_end=dur, nsteps=self.nsteps)
                U_f = res[-1]
                f_i += self.state_fidelity(U_f, self.U_t)
            fidelities.append(1 - (f_i / number_of_averages))
        
        fig = plt.figure(figsize=(11, 7))
        ax = fig.add_subplot(111)
        plt.subplots_adjust(bottom=0.15, top=0.9, right=0.98, left=0.1)
        plt.plot(microwave_noise_levels, fidelities, linewidth=1.5, zorder=10, label=r"Microwave Noise Fidelity")
        plt.grid(True, which="both")
        plt.xlabel(r'Microwave Noise', fontsize=14)
        plt.ylabel(r'FoM [a.u.]', fontsize=14)
        plt.legend()
        plt.savefig(os.path.join(self.results_path,"MWNoiseStability.pdf"), format="pdf", bbox_inches="tight")
        np.savetxt(os.path.join(self.results_path, 'MWNoiseStability_data.txt'), [microwave_noise_levels,fidelities])

        print("Analyzing 2D stability")

        # simulate 2d with rabi errors
        detunings = np.linspace(-7, 7, 30)
        rabiErrors = np.linspace(-0.5, 0.5, 30)
        start_time_2d = time.time()

        # Initialize the 2D array to store fidelities
        fidelities_2d = np.zeros((len(rabiErrors), len(detunings)))
        # Loop over all combinations of detunings and Rabi errors
        for i, rE in enumerate(rabiErrors):
            B1_pulse_amplitudes = np.array(B1_pulse[0])
            B1_pulse_times = B1_pulse[1]
            B1_pulse_rabi_amplitudes = B1_pulse_amplitudes * (1 + rE)
            B1_pulse_rabi = [B1_pulse_rabi_amplitudes.tolist(), B1_pulse_times]
            
            for j, det in enumerate(detunings):
                res = self.simulator.simulate(B1=B1_pulse_rabi, omega=self.H.transitions['|-3/2 -1/2> -> |-1/2 -1/2>'] + det, phi=0, RWA=False, progress_bar="", t_end=dur)
                U_f = res[-1]
                fidelities_2d[i, j] = 1 - self.state_fidelity(U_f, self.U_t) 
            print(f"{i+1}/{len(rabiErrors)}:{time.time()-start_time_2d} s. Remaining time: {(time.time()-start_time_2d)*(len(rabiErrors)-i-1)/(i+1)} s")

        vmin = fidelities_2d.min()
        vmax = fidelities_2d.max()

        tick_locations=([0.5,0.97,0.98,0.99,0.999] )
        # Plot the heatmap using pcolormesh
        fig, ax = plt.subplots(figsize=(11, 7))
        cax = ax.pcolormesh(detunings, rabiErrors, fidelities_2d, shading='auto', cmap='viridis', norm=SymLogNorm(linscale=0.001,linthresh=0.96,vmin=vmin, vmax=vmax, base=10))

        cbar = fig.colorbar(cax, ticks=tick_locations)
        cbar.set_label('Fidelity')

        plt.xlabel(r'Detuning $\Delta\omega$ [$\SI{}{\mega\hertz}$]', fontsize=14)
        plt.ylabel('Rabi Error [\%]', fontsize=14)
        plt.title('Fidelity Heatmap', fontsize=16)
        plt.grid(True, which="both")
        plt.savefig(os.path.join(self.results_path, "FidelityHeatmap.pdf"), format="pdf", bbox_inches="tight")
        np.savetxt(os.path.join(self.results_path, 'FidelityHeatmap_data.txt'), fidelities_2d)
        
        print("Analyzing the frequency")

        ft_pulse = np.concatenate([np.zeros(10000*len(pulse)), pulse, np.zeros(10000*len(pulse))])
        ft_timegrid = np.linspace(-10000*timegrid[-1], 10001*timegrid[-1], len(ft_pulse))

        pulse_ft_ampl = np.abs(np.fft.rfft(ft_pulse))
        pulse_ft_freq = np.fft.rfftfreq(len(ft_pulse), ft_timegrid[1]-ft_timegrid[0])
        # Plot the amplitude of the Fourier Transform
        fig = plt.figure(figsize=(11, 7))
        ax = fig.add_subplot(111)
        plt.subplots_adjust(bottom=0.15, top=0.9, right=0.98, left=0.1)
        plt.plot(pulse_ft_freq, pulse_ft_ampl, label=r'Pulse Fourier Transform')
        plt.xlabel(r'Frequency $f$ [\SI{}{\mega\hertz}]', fontsize=14)
        plt.ylabel(r'Amplitude', fontsize=14)
        plt.xlim(0,10)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.results_path,"FourierTransformPulse.pdf"), format="pdf", bbox_inches="tight")
        np.savetxt(os.path.join(self.results_path, 'FourierTransformPulse_data.txt'), [pulse_ft_freq,pulse_ft_ampl])

        #plt.show()

    def analyze_pulse(self, pulse1, pulse2=None, timegrid=None, dur=0.3, path="", save=False, 
                      simulator=None, H=None, analyze=[True, True, True, True, True, True, True, True]):
        """
        Analyse the pulse.

        Parameters:
        - pulse: The pulse.
        - timegrid: The time grid of the pulse.
        - dur: The duration of the pulse.

        Returns:
        None
        """
        plt.rcParams['text.usetex'] = True  # to use LaTeX in figures
        plt.rcParams["text.latex.preamble"]= r'\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage{lmodern, mathpazo}\inputencoding{utf8}\usepackage{amsmath}\usepackage{amssymb}\usepackage{dsfont}\usepackage{mathtools}\usepackage{physics}\usepackage{siunitx}\DeclareSIUnit\gauss{G}'
        plt.rcParams['font.family'] = ['serif']
        
        if analyze[0]:
            print("Plotting the pulses")
            dur_index = np.searchsorted(timegrid, dur, side='right') - 1
            # Plot the pulses over time
            fig = plt.figure(figsize=(11, 7))
            ax = fig.add_subplot(111)
            plt.step(timegrid[:dur_index], pulse1[:dur_index], linewidth=1.5, zorder=10, label=r"Optimized Pulse")
            if pulse2:
                plt.step(timegrid[:dur_index], pulse2[:dur_index], linewidth=1.5, zorder=10, linestyle='--', label=r"Square Pulse")
            plt.grid(True, which="both")
            plt.xlabel(r'Time $t$ [\SI{}{\micro\second}]', fontsize=22)
            plt.ylabel(r'Amplitude $B$ [\SI{}{\gauss}]', fontsize=22)
            plt.legend(fontsize=14)
            if save:
                plt.savefig(os.path.join(path,"PulseComparison.pdf"), format="pdf", bbox_inches="tight")

        if analyze[1]:
            print("Simulating the pulses")
            # Simulate and plot expectation values for Pulse 1
            states_to_plot = range(H.H.shape[0])
            states_to_plot = [0,1,2,3]
            B1_pulse1 = [pulse1, timegrid]
            result1 = simulator.simulate(B1=B1_pulse1, omega=H.transitions['|-3/2 -1/2> -> |-1/2 -1/2>'], phi=0, RWA=False, progress_bar="enhanced", t_end=dur)
            expectation_values1 = [qt.expect(H.get_state_dm(i), result1) for i in states_to_plot]

            if pulse2:
                # Simulate and plot expectation values for Pulse 2
                B1_pulse2 = [pulse2, timegrid]
                result2 = simulator.simulate(B1=B1_pulse2, omega=H.transitions['|-3/2 -1/2> -> |-1/2 -1/2>'], phi=0, RWA=False, progress_bar="enhanced", t_end=dur)
                expectation_values2 = [qt.expect(H.get_state_dm(i), result2) for i in states_to_plot]
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 7), sharey=True)
            else:
                fig, ax1 = plt.figure(figsize=(11, 7))
            #plt.subplots_adjust(bottom=0.15, top=0.9, right=0.98, left=0.1)
            ax1.set_xlabel(r'Time $t$ [\SI{}{\micro\second}]', fontsize=28)
            ax1.set_ylabel(r'Expectation values', fontsize=28)
            state_labels = Simulator.generate_state_labels(H.H.dims[0])
            artists1 = []
            artists2 = []
            color_list = [mittelblau, hellblau, '#1e9c89', anthrazit] 
            for i in states_to_plot:
                # artist, = ax1.plot(simulator.tlist, expectation_values1[i], label=f'{state_labels[i]}', color=color_list[i])
                artist, = ax1.plot(simulator.tlist, expectation_values1[i], label=f'{state_labels[i]}')
                artists1.append(artist)
                if pulse2:
                    #artist2, =ax2.plot(simulator.tlist, expectation_values2[i], linestyle='--', label=f'{state_labels[i]}', color=color_list[i])
                    artist2, =ax2.plot(simulator.tlist, expectation_values2[i], linestyle='--', label=f'{state_labels[i]}')
                    artists2.append(artist2)
            
            if pulse2:
                ax2.set_xlabel(r'Time $t$ [\SI{}{\micro\second}]', fontsize=28)
                ax2.grid(True, which="both")
                ax2.set_title(r"Square Pulse", fontsize=28)
                ax2.legend(handles=artists1+artists2, loc='upper right', bbox_to_anchor=(0.04,1),fontsize=18)
            ax1.grid(True, which="both")
            ax1.legend(handles=artists1+artists2, loc='upper left', bbox_to_anchor=(0.96,1),fontsize=18)
            ax1.set_title(r"Optimized Pulse", fontsize=28)
            ax2.tick_params(axis='both', which='major', labelsize=18)
            ax1.tick_params(axis='both', which='major', labelsize=18)  # You can adjust the size value
            if save:
                plt.savefig(os.path.join(path,"SimulationComparison.pdf"), format="pdf", bbox_inches="tight")

        if analyze[2]:
            print("Analyzing the detuning stability")
            detunings = np.linspace(-15,15,200)
            fidelities1 = []
            start_time_det = time.time()
            if pulse2:
                fidelities2 = []
            for i,det in enumerate(detunings):
                res1 = simulator.simulate(B1=[pulse1,timegrid], omega=self.H.transitions['|-3/2 -1/2> -> |-1/2 -1/2>']+det, phi=0, RWA=False, progress_bar="", t_end=dur)
                U_f1 = res1[-1]
                fidelities1.append(1-self.state_fidelity(U_f1,self.U_t))

                if pulse2:
                    res2 = simulator.simulate(B1=[pulse2,timegrid], omega=self.H.transitions['|-3/2 -1/2> -> |-1/2 -1/2>']+det, phi=0, RWA=False, progress_bar="", t_end=dur)
                    U_f2 = res2[-1]
                    fidelities2.append(1-self.state_fidelity(U_f2,self.U_t))
                print(f"{i+1}/{len(detunings)}:{time.time()-start_time_det} s. Remaining time: {(time.time()-start_time_det)*(len(detunings)-i-1)/(i+1)} s")
            fig = plt.figure(figsize=(11, 7))
            ax = fig.add_subplot(111)
            plt.plot(detunings, fidelities1, linewidth=1.5, zorder=10, label=r"Optimized Pulse", color=mittelblau)
            if pulse2:
                plt.plot(detunings, fidelities2, linewidth=1.5, zorder=10, label=r"Square Pulse", color=hellblau)
            x_barrier = 8.61564484e+00 / 2
            ax.axvline(x=x_barrier, color=anthrazit, linestyle='--', linewidth=1.5, label=r"$A_{zz}/2$")
            plt.grid(True, which="both")
            plt.xlabel(r'Detuning $\Delta\omega$ [$\SI{}{\mega\hertz}$]', fontsize=28)
            plt.ylabel(r'Fidelity', fontsize=28)
            plt.tick_params(axis='both', which='major', labelsize=18)  # You can adjust the size value
            plt.legend(fontsize=18)
            if save:
                plt.savefig(os.path.join(path,"DetuningStability.pdf"), format="pdf", bbox_inches="tight")
                np.savetxt(os.path.join(path, 'DetuningStability1_data.txt'), [detunings,fidelities1])
                if pulse2:
                    np.savetxt(os.path.join(path, 'DetuningStability2_data.txt'), [detunings,fidelities2])
        
        if analyze[3]:
            print("Analyzing the Rabi Error stability")
            rabiErrors = np.linspace(-1, 1, 200)
            fidelities1 = []
            if pulse2:
                fidelities2 = []
            start_time_rabi = time.time()
            for i, rE in enumerate(rabiErrors):
                B1_pulse1_rabi_amplitudes = np.array(pulse1) * (1 + rE)
                B1_pulse1_rabi = [B1_pulse1_rabi_amplitudes.tolist(), timegrid]
                
                res1 = simulator.simulate(B1=B1_pulse1_rabi, omega=H.transitions['|-3/2 -1/2> -> |-1/2 -1/2>'], phi=0, RWA=False, progress_bar="", t_end=dur)
                U_f1 = res1[-1]
                fidelities1.append(1-self.state_fidelity(U_f1, self.U_t))
                
                if pulse2:
                    B1_pulse2_rabi_amplitudes = np.array(pulse2) * (1 + rE)
                    B1_pulse2_rabi = [B1_pulse2_rabi_amplitudes.tolist(), timegrid]
                    
                    res2 = simulator.simulate(B1=B1_pulse2_rabi, omega=H.transitions['|-3/2 -1/2> -> |-1/2 -1/2>'], phi=0, RWA=False, progress_bar="", t_end=dur)
                    U_f2 = res2[-1] 
                    fidelities2.append(1-self.state_fidelity(U_f2, self.U_t))
                print(f"{i+1}/{len(rabiErrors)}:{time.time()-start_time_rabi} s. Remaining time: {(time.time()-start_time_rabi)*(len(rabiErrors)-i-1)/(i+1)} s")

            fig = plt.figure(figsize=(11, 7))
            ax = fig.add_subplot(111)
            plt.plot(rabiErrors, fidelities1, linewidth=1.5, zorder=10, label=r"Optimized Pulse", color=mittelblau)
            if pulse2:
                plt.plot(rabiErrors, fidelities2, linewidth=1.5, zorder=10, label=r"Square Pulse", color=hellblau)
            plt.grid(True, which="both")
            plt.xlabel(r'Rabi Error [$B^{-1}_\mathrm{mw}$]', fontsize=28)
            plt.ylabel(r'Fidelity', fontsize=28)
            plt.tick_params(axis='both', which='major', labelsize=18)  # You can adjust the size value
            plt.legend(fontsize=18, loc='lower center')
            if save:
                plt.savefig(os.path.join(path,"RabiStability.pdf"), format="pdf", bbox_inches="tight")
                np.savetxt(os.path.join(path, 'RabiStability1_data.txt'), [rabiErrors,fidelities1])
                if pulse2:
                    np.savetxt(os.path.join(path, 'RabiStability2_data.txt'), [rabiErrors,fidelities2])

        if analyze[6]:
            print("Analyzing the Duration Error stability")
            durErrors = np.linspace(-1, 1, 200)
            fidelities1 = []
            if pulse2:
                fidelities2 = []
            start_time_dur = time.time()
            for i, dE in enumerate(durErrors):
                B1_pulse1_dur_timegrid = np.array(timegrid) * (1 + dE) 
                B1_pulse1_dur = [pulse1, B1_pulse1_dur_timegrid]
                
                res1 = simulator.simulate(B1=B1_pulse1_dur, omega=H.transitions['|-3/2 -1/2> -> |-1/2 -1/2>'], phi=0, RWA=False, progress_bar="", t_end=dur*(1 + dE))
                U_f1 = res1[-1]
                fidelities1.append(1-self.state_fidelity(U_f1, self.U_t))
                
                if pulse2:
                    B1_pulse2_dur_timegrid = np.array(timegrid) * (1 + dE)
                    B1_pulse2_dur = [pulse2, B1_pulse2_dur_timegrid]
                    
                    res2 = simulator.simulate(B1=B1_pulse2_dur, omega=H.transitions['|-3/2 -1/2> -> |-1/2 -1/2>'], phi=0, RWA=False, progress_bar="", t_end=dur*(1 + dE))
                    U_f2 = res2[-1] 
                    fidelities2.append(1-self.state_fidelity(U_f2, self.U_t))
                print(f"{i+1}/{len(durErrors)}:{time.time()-start_time_dur} s. Remaining time: {(time.time()-start_time_dur)*(len(durErrors)-i-1)/(i+1)} s")
           
            fig = plt.figure(figsize=(11, 7))
            ax = fig.add_subplot(111)
            plt.plot(durErrors, fidelities1, linewidth=1.5, zorder=10, label=r"Optimized Pulse", color=mittelblau)
            if pulse2:
                plt.plot(durErrors, fidelities2, linewidth=1.5, zorder=10, label=r"Square Pulse", color=hellblau)
            plt.grid(True, which="both")
            plt.xlabel(r'Duration Error [$t^{-1}_\mathrm{dur}$]', fontsize=28)
            plt.ylabel(r'Fidelity', fontsize=28)
            plt.tick_params(axis='both', which='major', labelsize=18)  # You can adjust the size value
            plt.legend(fontsize=18, loc='lower center')
            if save:
                plt.savefig(os.path.join(path,"DurStability.pdf"), format="pdf", bbox_inches="tight")
                np.savetxt(os.path.join(path, 'DurStability1_data.txt'), [durErrors,fidelities1])
                if pulse2:
                    np.savetxt(os.path.join(path, 'DurStability2_data.txt'), [durErrors,fidelities2])
        
        if analyze[7]:
            print("Analyzing the Microwave Noise stability")
            number_of_averages = 10
            microwave_noise_levels = np.linspace(0, 0.5, 300)
            fidelities1 = []
            if pulse2:
                fidelities2 = []
            start_time_noise = time.time()
            for i, nE in enumerate(microwave_noise_levels):
                f_i1 = 0
                if pulse2:
                    f_i2 = 0
                for j in range(number_of_averages):
                    B1_pulse1_noise = [(np.array(pulse1) + np.random.normal(0, nE, len(pulse1))).tolist(), timegrid]
                    res1 = simulator.simulate(B1=B1_pulse1_noise, omega=H.transitions['|-3/2 -1/2> -> |-1/2 -1/2>'], phi=0, RWA=False, progress_bar="", t_end=dur)
                    U_f1 = res1[-1]
                    f_i1 += (1-self.state_fidelity(U_f1, self.U_t))
                    
                    if pulse2:
                        B1_pulse2_noise = [(np.array(pulse2) + np.random.normal(0, nE, len(pulse2))).tolist(), timegrid]
                        
                        res2 = simulator.simulate(B1=B1_pulse2_noise, omega=H.transitions['|-3/2 -1/2> -> |-1/2 -1/2>'], phi=0, RWA=False, progress_bar="", t_end=dur)
                        U_f2 = res2[-1] 
                        
                        f_i2 += (1-self.state_fidelity(U_f2, self.U_t))
                fidelities1.append(f_i1/number_of_averages)
                fidelities2.append(f_i2/number_of_averages)
                print(f"{i+1}/{len(microwave_noise_levels)}:{time.time()-start_time_noise} s. Remaining time: {(time.time()-start_time_noise)*(len(microwave_noise_levels)-i-1)/(i+1)} s")
           
            fig = plt.figure(figsize=(11, 7))
            ax = fig.add_subplot(111)
            plt.plot(microwave_noise_levels, fidelities1, linewidth=1.5, zorder=10, label=r"Optimized Pulse", color=mittelblau)
            if pulse2:
                plt.plot(microwave_noise_levels, fidelities2, linewidth=1.5, zorder=10, label=r"Square Pulse", color=hellblau)
            plt.grid(True, which="both")
            plt.xlabel(r'Standard deviation $\sigma$ [$\SI{}{\mathrm{G}}$]', fontsize=28)
            plt.ylabel(r'Fidelity', fontsize=28)
            plt.tick_params(axis='both', which='major', labelsize=18)  # You can adjust the size value
            plt.legend(fontsize=18, loc='lower center')
            if save:
                plt.savefig(os.path.join(path,f"NoiseStability{number_of_averages}averages.pdf"), format="pdf", bbox_inches="tight")
                np.savetxt(os.path.join(path, 'NoiseStability1_data.txt'), [microwave_noise_levels,fidelities1])
                if pulse2:
                    np.savetxt(os.path.join(path, 'NoiseStability2_data.txt'), [microwave_noise_levels,fidelities2])
           
        if analyze[4]:
            print("Analyzing 2D stability")
            # simulate 2d with rabi errors for both pulses
            detunings = np.linspace(-5, 5, 45)
            rabiErrors = np.linspace(-0.5, 0.5, 45)
            start_time_2d = time.time()
            fidelities_2d_1 = np.zeros((len(rabiErrors), len(detunings)))
            if pulse2:
                fidelities_2d_2 = np.zeros((len(rabiErrors), len(detunings)))

            for i, rE in enumerate(rabiErrors):
                B1_pulse1_rabi_amplitudes = np.array(pulse1) * (1 + rE)
                B1_pulse1_rabi = [B1_pulse1_rabi_amplitudes.tolist(), timegrid]
                
                if pulse2:
                    B1_pulse2_rabi_amplitudes = np.array(pulse2) * (1 + rE)
                    B1_pulse2_rabi = [B1_pulse2_rabi_amplitudes.tolist(), timegrid]
                
                for j, det in enumerate(detunings):
                    res1 = simulator.simulate(B1=B1_pulse1_rabi, omega=H.transitions['|-3/2 -1/2> -> |-1/2 -1/2>'] + det, phi=0, RWA=False, progress_bar="", t_end=dur)
                    U_f1 = res1[-1]
                    fidelities_2d_1[i, j] = 1-self.state_fidelity(U_f1, self.U_t, compare_target_states=False)
                    
                    if pulse2:
                        res2 = simulator.simulate(B1=B1_pulse2_rabi, omega=H.transitions['|-3/2 -1/2> -> |-1/2 -1/2>'] + det, phi=0, RWA=False, progress_bar="", t_end=dur)
                        U_f2 = res2[-1]
                        fidelities_2d_2[i, j] = 1-self.state_fidelity(U_f2, self.U_t, compare_target_states=False)
                print(f"{i+1}/{len(rabiErrors)}:{time.time()-start_time_2d} s. Remaining time: {(time.time()-start_time_2d)*(len(rabiErrors)-i-1)/(i+1)} s")
            if pulse2:
                # Find common vmin and vmax for color scales
                vmin = max(0,min(fidelities_2d_1.min(), fidelities_2d_2.min()))
                vmax = max(fidelities_2d_1.max(), fidelities_2d_2.max())

                #generate logarithmic ticks
                tick_locations=([0.36,0.56,0.76, 0.96,0.97,0.98,0.99,0.999] )

                # Plot the heatmaps side by side
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 7), sharey=True)
            else: 
                fig, ax1 = plt.subplots(1, 1, figsize=(11, 7))

            cax1 = ax1.pcolormesh(detunings, rabiErrors, fidelities_2d_1, shading='auto', cmap='viridis', norm=SymLogNorm(linscale=0.01,linthresh=0.96,vmin=vmin, vmax=vmax, base=10))
            ax1.set_xlabel(r'Detuning $\Delta\omega$ [$\SI{}{\mega\hertz}$]', fontsize=28)
            ax1.set_ylabel(r'Rabi Error [$B^{-1}_\mathrm{mw}$]', fontsize=28)
            #ax1.set_title(r'Fidelity Heatmap for Pulse 1', fontsize=16)
            #ax1.grid(True, which="both")
            
            if pulse2:
                cax2 = ax2.pcolormesh(detunings, rabiErrors, fidelities_2d_2, shading='auto', cmap='viridis', norm=SymLogNorm(linscale=0.01,linthresh=0.96,vmin=vmin, vmax=vmax, base=10))
                ax2.set_xlabel(r'Detuning $\Delta\omega$ [$\SI{}{\mega\hertz}$]', fontsize=28)
                #ax2.set_title(r'Fidelity Heatmap for Pulse 2', fontsize=16)
                #ax2.grid(True, which="both")

                # Add a single colorbar for both heatmaps
                cbar = fig.colorbar(cax2, ax=(ax1,ax2),ticks=tick_locations)
            else:
                cbar = fig.colorbar(cax1,ticks=tick_locations)
            cbar.set_label(r'Fidelity', fontsize=28)
            ax1.tick_params(axis='both', which='major', labelsize=18) 
            ax2.tick_params(axis='both', which='major', labelsize=18)  
            cbar.ax.tick_params(axis='both', which='major', labelsize=18)  

            if save:
                plt.savefig(os.path.join(path, "FidelityHeatmapComparison.pdf"), format="pdf", bbox_inches="tight")
                np.savetxt(os.path.join(path, 'FidelityHeatmap_Pulse1_data.txt'), fidelities_2d_1)
                if pulse2:
                    np.savetxt(os.path.join(path, 'FidelityHeatmap_Pulse2_data.txt'), fidelities_2d_2)

        if analyze[5]:
            print("Analyzing the frequency")
            # Fourier transform comparison
            ft_pulse1 = np.concatenate([np.zeros(10000*len(pulse1)), pulse1, np.zeros(10000*len(pulse1))])
            if pulse2:
                ft_pulse2 = np.concatenate([np.zeros(10000*len(pulse2)), pulse2, np.zeros(10000*len(pulse2))])
            ft_timegrid = np.linspace(-10000*timegrid[-1], 10001*timegrid[-1], len(ft_pulse1))

            pulse_ft_ampl1 = np.abs(np.fft.rfft(ft_pulse1))
            pulse_ft_freq1 = np.fft.rfftfreq(len(ft_pulse1), ft_timegrid[1]-ft_timegrid[0])

            if pulse2:
                pulse_ft_ampl2 = np.abs(np.fft.rfft(ft_pulse2))
                pulse_ft_freq2 = np.fft.rfftfreq(len(ft_pulse2), ft_timegrid[1]-ft_timegrid[0])

            fig = plt.figure(figsize=(11, 7))
            ax = fig.add_subplot(111)
            plt.plot(pulse_ft_freq1, pulse_ft_ampl1, label=r'Optimized Pulse', color=mittelblau)
            if pulse2:
                plt.plot(pulse_ft_freq2, pulse_ft_ampl2, label=r'Square Pulse', color=hellblau)
            plt.xlabel(r'Frequency $f$ [\SI{}{\mega\hertz}]', fontsize=28)
            plt.ylabel(r'Amplitude [a.U.]', fontsize=28)
            plt.xlim(0,10)
            plt.legend(fontsize=18)
            plt.tick_params(axis='both', which='major', labelsize=18)  # You can adjust the size value
            plt.grid(True)
            if save:
                plt.savefig(os.path.join(path,"FourierTransformPulseComparison.pdf"), format="pdf", bbox_inches="tight")
                np.savetxt(os.path.join(path, 'FourierTransformPulseComparison_data.txt'), np.vstack((pulse_ft_freq1, pulse_ft_ampl1, pulse_ft_ampl2)))

        plt.show()

    def read_pulse_data(self,folder_name):
        """
        Read pulse data from files in the specified folder.
        
        Parameters:
        - folder_name (str): The path to the folder containing the pulse data files.
        
        Returns:
        - pulse (ndarray): An array of floats representing the pulse data.
        - timegrid (ndarray): An array of floats representing the time grid.
        - dur (float): The duration of the pulse.
        """
        # Define file paths
        best_controls_path = os.path.join(folder_name, "bestControls.txt")
        best_params_path = os.path.join(folder_name, "bestParams.txt")

        # Initialize variables
        pulse = None
        timegrid = None
        dur = None
        
        # Read bestControls.txt
        with open(best_controls_path, 'r') as f:
            lines = f.readlines()
            pulse = np.array([float(x) for x in lines[0].split()])
            timegrid = np.array([float(x) for x in lines[1].split()])
        
        # Read bestParams.txt
        with open(best_params_path, 'r') as f:
            dur = float(f.readline().strip())
        
        return pulse, timegrid, dur
    
    def transmission_pulse(self,pulse, timegrid, frequencies, transmissions, cutoff_freq=500):
        """
        Reconstructs a pulse signal by applying transmission correction in the frequency domain.
        
        Parameters:
            pulse (array-like): The input pulse signal.
            timegrid (array-like): The time grid of the pulse signal.
            frequencies (array-like): The frequencies corresponding to the transmission data.
            transmissions (array-like): The transmission values corresponding to the frequencies.
            cutoff_freq (float, optional): The cutoff frequency for transmission correction. Frequencies above this value will be set to 1.0. Defaults to 500.
        
        Returns:
            array-like: The reconstructed pulse signal after applying transmission correction.
        """
        ft_pulse = np.concatenate([np.zeros(10000*len(pulse)), pulse, np.zeros(10000*len(pulse))])
        ft_timegrid = np.linspace(-10000*timegrid[-1], 10001*timegrid[-1], len(ft_pulse))

        #transform the pulse to frequency domain keeping the complex values for the phase
        pulse_ft_complex = np.fft.rfft(ft_pulse) 
        pulse_ft_freq = np.fft.rfftfreq(len(ft_pulse), ft_timegrid[1]-ft_timegrid[0])

        # Interpolate the transmission data to match the Fourier transform frequencies
        interpolation_function =  interp1d(frequencies, transmissions, bounds_error=False, fill_value=1.0)
        #interpolation_function = CubicSpline(frequencies, transmissions, extrapolate=False)

        transmission = interpolation_function(pulse_ft_freq)
        if cutoff_freq != -1:
            transmission[pulse_ft_freq > cutoff_freq] = 1.0


        adjusted_pulse_ft = pulse_ft_complex / transmission

        reconstructed_pulse = np.fft.irfft(adjusted_pulse_ft)

        # Extract the original pulse portion (removing the padded zeros)
        start_index = 10000 * len(pulse)
        end_index = start_index + len(pulse)
        reconstructed_pulse = reconstructed_pulse[start_index:end_index]
        return reconstructed_pulse
    
    def adjust_sampling_rate(self, pulse, timegrid, sampling_rate=12e3):
        """
        Adjust the sampling rate of a pulse signal by interpolating the pulse values.

        Parameters:
        - pulse (ndarray): The pulse signal.
        - timegrid (ndarray): The time grid of the pulse signal.
        - sampling_rate (float): The desired sampling rate.

        Returns:
        - ndarray: The new pulse signal with the adjusted sampling rate.
        - ndarray: The new time grid with the adjusted sampling rate.
        """
        duration = timegrid[-1]  # Total duration (in microseconds)
        num_samples = int(sampling_rate * duration)  # Number of samples for the new array

        # New time array with the desired sampling rate
        new_timegrid = np.linspace(timegrid[0], timegrid[-1], num_samples)

        # Create a new pulse array by copying the old amplitude values until the next timegrid point
        new_pulse = np.zeros(num_samples)

        # Iterate over the new timegrid to assign pulse values
        old_index = 0
        for i in range(num_samples):
            # While the new time is greater than the current timegrid point, move to the next one
            while old_index < len(timegrid) - 1 and new_timegrid[i] >= timegrid[old_index + 1]:
                old_index += 1
            # Copy the current pulse value
            new_pulse[i] = pulse[old_index]
        return new_pulse, new_timegrid


            
    
    


        

