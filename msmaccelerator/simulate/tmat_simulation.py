"""OpenMM simulation device. We connect to server, request a starting
structure, and then propagate.
"""
#############################################################################
# Imports
##############################################################################

import os

from IPython.utils.traitlets import CInt
import mdtraj
import scipy.io

import numpy as np

from ..core.device import Device
from ..core.traitlets import FilePath


# local
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

    gens_fn = FilePath('Gens.h5', config=True, help='''
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
    gens = None

    def start(self):
        # Load transition matrix
        t_matrix = scipy.io.mmread(self.tmat_fn)
        t_matrix = t_matrix.tocsr()
        self.t_matrix = t_matrix
        self.log.info('Loaded transition matrix of shape %s',
                      self.t_matrix.shape)

        # Load generators
        self.gens = mdtraj.load(self.gens_fn)

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
        self.log.info('Starting TMat simulation...')

        # Get that starting state path
        starting_state_path = content.starting_state.path
        assert content.output.protocol == 'localfs', "I'm currently only equipped for localfs output"

        # Parse starting state
        with open(starting_state_path) as f:
            line = f.readline()
            line = line.strip()
            state_i = int(line)

        # Initialize output array
        xyz = np.zeros((self.number_of_steps // self.report_interval,
                        self.gens.n_atoms, 3))

        report = self.report_interval
        report_toti = 0
        for _ in xrange(self.number_of_steps):
            # Get stuff from our sparse matrix
            t_matrix = self.t_matrix
            csr_slicer = slice(t_matrix.indptr[state_i], t_matrix.indptr[state_i + 1])
            probs = t_matrix.data[csr_slicer]
            colinds = t_matrix.indices[csr_slicer]

            # Check for normalization
            np.testing.assert_almost_equal(np.sum(probs), 1,
                                  err_msg="TMatrix isn't row normalized.")

            # Find our new state and translate to actual indices
            prob_i = np.sum(np.cumsum(probs) < np.random.rand())
            state_i = colinds[prob_i]

            # Check to see if we report
            if report == self.report_interval:
                xyz[report_toti] = self.gens.xyz[state_i]

                # Reset
                report = 0
                report_toti += 1

            report += 1

        # Write
        assert report_toti == xyz.shape[0], "I did my math wrong."
        out_traj = mdtraj.Trajectory(xyz=xyz, topology=self.gens.topology)
        out_traj.save(content.output.path)
        
        self.log.info('Finished TMat simulation.')

        # Say that we're done
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
