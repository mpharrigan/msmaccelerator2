"""OpenMM simulation device. We connect to server, request a starting
structure, and then propagate.
"""
#############################################################################
# Imports
##############################################################################

import os
import sys
import numpy as np
from IPython.utils.traitlets import Unicode, CInt, Instance, Bool, Enum
from mdtraj.reporters import HDF5Reporter
import simtk.openmm as mm
from simtk.openmm import XmlSerializer, Platform
from simtk.openmm.app import (Simulation, PDBFile)
from ..core.traitlets import FilePath

import scipy.io

# local
from .reporters import CallbackReporter
from ..core.device import Device

#############################################################################
# Handlers
##############################################################################


class TMatSimulator(Device):
    name = 'TMat'
    path = 'msmaccelerator.simulate.tmat_simulation.TMatSimulator'
    short_description = 'Sample a single round from a transition matrix'
    long_description = '''This device will connect to the msmaccelerator server,
        request the initial conditions with which to start a simulation, and
        simulate the propogation of dynamics by performing kinetic monte
        carlo from a pre-existing transition matrix.'''

    # configurables.
    tmat_fn = FilePath('tProb.mtx', config=True, help='''
        Path to the transition matrix from which to sample.''')
    
    gens_fn = FilePath('Gens.lh5', config=True, help='''
        Path to the generators trajectory.''')

    number_of_steps = CInt(10000, config=True, help='''
        Number of steps of dynamics to do''')

    report_interval = CInt(1000, config=True, help='''
        Interval at which to save positions to a disk, in units of steps''')

    


    # expose these as command line flags on --help
    # other settings can still be specified on the command line, its just
    # less convenient
    aliases = dict(tmat_fn='TMatSimulator.tmat_fn',
                  number_of_steps='TMatSimulator.number_of_steps',
                  report_interval='TMatSimulator.report_interval')

    t_matrix = None

    def start(self):
        self.t_matrix = scipy.io.mmread(self.tmat_fn)
        self.log.info('Shape! {}'.format(self.t_matrix.shape))
        super(TMatSimulator, self).start()

    def on_startup_message(self, msg):
        """This method is called when the device receives its startup message
        from the server.
        """

        assert msg.header.msg_type in ['simulate']  # only allowed RPC
        return getattr(self, msg.header.msg_type)(msg.header, msg.content)

    def simulate(self, header, content):
        """Main method that is "executed" by the receipt of the
        msg_type == 'simulate' message from the server.

        We run some KMC dynamics, and then send back the results.
        """
        self.log.info('Setting up TMat simulation...')
        
#         state, topology = self.deserialize_input(content)
        starting_state_path = content.starting_state.path
# 
#         # set the GPU platform
#         platform = Platform.getPlatformByName(str(self.platform))
#         if self.platform == 'CUDA':
#             properties = {'CudaPrecision': 'mixed',
#                           'CudaDeviceIndex': str(self.device_index)
#                          }
#         elif self.platform == 'OpenCL':
#             properties = {'OpenCLPrecision': 'mixed',
#                           'OpenCLDeviceIndex': str(self.device_index)
#                          }
#         else:
#             properties = None
# 
# 
#         simulation = Simulation(topology, self.system, self.integrator,
#                                 platform, properties)
#         # do the setup
#         self.set_state(state, simulation)
#         self.sanity_check(simulation)
#         if self.minimize:
#             self.log.info('minimizing...')
#             simulation.minimizeEnergy()
# 
#         if self.random_initial_velocities:
#             try:
#                 temp = simulation.integrator.getTemperature()
#                 simulation.context.setVelocitiesToTemperature(temp)
#             except AttributeError:
#                 print "I don't know what temperature to use!!"
#                 # TODO: look through the system's forces to find an andersen
#                 # thermostate?
#                 raise
#             pass

        assert content.output.protocol == 'localfs', "I'm currently only equiped for localfs output"
#         self.log.info('adding reporters...')
#         self.add_reporters(simulation, content.output.path)
# 
#         # run dynamics!
#         self.log.info('Starting dynamics')
#         simulation.step(self.number_of_steps)
# 
#         for reporter in simulation.reporters:
#             # explicitly delete the reporters so that any open file handles
#             # are closed.
#             del reporter

        # tell the master that I'm done
        self.send_recv(msg_type='simulation_done', content={
            'status': 'success',
            'starting_state': {
                    'protocol': 'localfs',
                    'path': starting_state_path
            },
            'output': {
                'protocol': 'localfs',
                'path': content.output.path
            }
        })

    ##########################################################################
    # Begin helpers for setting up the simulation
    ##########################################################################

    def sanity_check(self, simulation):
        positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
        for atom1, atom2 in simulation.topology.bonds():
            d = np.linalg.norm(positions[atom1.index, :] - positions[atom2.index, :])
            if not d < 0.3:
                self.log.error(positions[atom1.index, :])
                self.log.error(positions[atom2.index, :])
                raise ValueError('atoms are bonded according to topology but not close by '
                                 'in space: %s. %s' % (d, positions))


    def deserialize_input(self, content):
        """Retreive the state and topology from the message content

        The message protocol tries not to pass 'data' around within the
        messages, but instead pass paths to data. So far we're only sending
        paths on the local filesystem, but we might could generalize this to
        HTTP or S3 or something later.

        The assumption that data can be passed around on the local filesystem
        shouldn't be built deep into the code at all
        """
        # todo: better name for this function?

        if content.starting_state.protocol == 'localfs':
            with open(content.starting_state.path) as f:
                self.log.info('Opening state file: %s', content.starting_state.path)
                state = XmlSerializer.deserialize(f.read())
        else:
            raise ValueError('Unknown protocol')

        if content.topology_pdb.protocol == 'localfs':
            topology = PDBFile(content.topology_pdb.path).topology
        else:
            raise ValueError('Unknown protocol')

        return state, topology

    def set_state(self, state, simulation):
        "Set the state of a simulation to whatever is in the state object"
        # why do I have to do this so... manually?
        # this is why:

        # simulation.context.setState(state)
        # TypeError: in method 'Context_setState', argument 2 of type 'State const &'

        simulation.context.setPositions(state.getPositions())
        simulation.context.setVelocities(state.getVelocities())
        simulation.context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
        for key, value in state.getParameters():
            simulation.context.setParameter(key, value)

    def add_reporters(self, simulation, outfn):
        "Add reporters to a simulation"
        def reporter_callback(report):
            """Callback for processing reporter output"""
            self.log.info(report)

        callback_reporter = CallbackReporter(reporter_callback,
            self.report_interval, step=True, potentialEnergy=True,
            temperature=True, time=True, total_steps=self.number_of_steps)
        h5_reporter = HDF5Reporter(outfn, self.report_interval, coordinates=True,
                                   time=True, cell=True, potentialEnergy=True,
                                   kineticEnergy=True, temperature=True)

        simulation.reporters.append(callback_reporter)
        simulation.reporters.append(h5_reporter)


###############################################################################
# Utilities
###############################################################################

def random_seed():
    """Get a seed for a random number generator, based on the current platform,
    pid, and wall clock time.

    Returns
    -------
    seed : int
        The seed is a 32-bit int
    """
    import platform
    import time
    import hashlib
    plt = ''.join(platform.uname())
    seed = int(hashlib.md5('%s%s%s' % (plt, os.getpid(), time.time())).hexdigest(), 16)

    return seed % np.iinfo(np.int32).max
