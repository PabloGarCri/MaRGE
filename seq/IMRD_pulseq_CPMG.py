import pypulseq as pp

import os
import sys

from configs.hw_config import grad_raster_time, grad_rise_time

#*****************************************************************************
# Get the directory of the current script
main_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(main_directory)
parent_directory_imrdparams = os.path.join(parent_directory,'IMRD_Parameters')
parent_directory = os.path.dirname(parent_directory)

# Define the subdirectories you want to add to sys.path
subdirs = ['MaRGE', 'marcos_client']

# Add the subdirectories to sys.path
for subdir in subdirs:
    full_path = os.path.join(parent_directory, subdir)
    sys.path.append(full_path)
#******************************************************************************
import numpy as np
import controller.experiment_gui as ex
import configs.hw_config as hw  # Import the scanner hardware config
import configs.units as units
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
from marga_pulseq.interpreter import PSInterpreter  # Import the flocra-pulseq interpreter
import pypulseq as pp  # Import PyPulseq
import scipy.signal as scp


# Template Class for MRI Sequences
class IMRD(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        """
        Defines the parameters for the sequence.

        Instructions for students:
        - Each parameter is defined using the `addParameter` method, which takes the following arguments:
          - key (str): A unique identifier for the parameter. This will be used to reference the value in other parts of the sequence.
          - string (str): A human-readable description of the parameter. This should clearly describe what the value represents.
          - val (int/float/str/list): The default value for the parameter. It can be a number, string, or list depending on the requirement.
          - units (optional): Units associated with the parameter (e.g., ms, cm, etc.). Use the `configs.units` module for common units.
          - field (str): The category to which the parameter belongs. It can be:
            - 'RF': Radio Frequency (parameters related to RF pulses).
            - 'IM': Imaging (parameters related to image acquisition).
            - 'SEQ': Sequence (parameters related to the sequence structure).
            - 'OTH': Other parameters that don't fit into the above categories.
          - tip (optional): Additional tips or information about the parameter, such as recommendations or constraints.
        """
        super(IMRD, self).__init__()

        # Sequence name (Do not include 'field')
        self.addParameter(key='seqName', string='Sequence Name', val='IMRD_CPMG',
                          tip="The identifier name for the sequence.")

        self.addParameter(key='toMaRGE', val=True)

        # Number of scans
        self.addParameter(key='nScans', string='Number of scans', val=1, field='IM',
                          tip="Number of repetitions of the full scan.")
        
        #Delay between scans
        self.addParameter(key='delayScans', string='Delay between scans', val=1, field='IM',
                          tip="Delay between repetitions of the full scan in miliseconds.")
        
        # Acquisition bandwidth
        self.addParameter(key='nPoints', string='Number of points', val=200, field='IM',
                          tip="Number of points per individual repetition acquisition.")

        # Excitation time
        self.addParameter(key='rfExTime', string='Excitation Pulse Duration (us)', val=100.0, units=units.us,
                          field='RF',
                          tip="Duration of the RF excitation pulse in microseconds (us).")

        self.addParameter(key='file', string='Paramter File', val='test.txt', field='SEQ', tip="Path to the .txt file containing the FAs and TRs")

        self.addParameter(key='shimming', string='Shimming', val=[0.0, 0.0, 0.0], field='SEQ', units=units.sh)

        self.addParameter(key='spoke', string='Spokes', val=0, field='SEQ',tip=' 1 = 1D, 0 = Gradientless')

        self.addParameter(key='spokeAxis', string='Axis for Spokes', val='x', field='SEQ')

        self.addParameter(key='piPulseType', string='Type of refocusing pulses', val='CP', field='RF')

        self.addParameter(key='nEchos', string='Number of echoes', val=3, field='RF')

        self.addParameter(key='acquistionTime', string = 'Acquistion Time', val = 4.0 , field = 'IM' )

        self.rotation = np.zeros(4)
        self.angle = None
        self.rotationAxis = None

    def sequenceInfo(self):
        """
        Description of the sequence. Students should customize this.
        """
        print("IMRD sequence with fixed windows and TSE acquisition")
        print("Pablo García-Cristóbal")
        print("i3M, UPV-CSIC, València, Spain\n")
        print(" ><(((º> ")

    def sequenceTime(self):
        """
        Calculate the sequence time based on its parameters.
        Students can extend this method as needed.
        """
        nScans = self.mapVals['nScans']
        params = np.array(np.loadtxt(self.mapVals['file']))
        nRepetitions = int((len(params) - 1 ) / 2)
        TRs = params[0:1 + nRepetitions] * 1e-3  # s
        repetitionTime= sum(TRs)
        delayScans= self.mapVals['delayScans']
        seqTime = nScans * (repetitionTime + delayScans) / 60  # conversion to minutes
        seqTime = np.round(seqTime, decimals=2)
        return seqTime  # minutes

    def sequenceRun(self, plot_seq=False, demo=False, standalone=False):
        """
        Run the MRI sequence.

        This method initializes batches and creates the full sequence by iterating through slices and phase-encoding steps.

        Instructions for students:
        - Batches divide the sequence into smaller, hardware-manageable sections.
        - `initializeBatch` sets up new sequence batches, while `createBatches` iterates through slices and phase-encoding to create the full sequence.

        Args:
            plotSeq (bool): If True, plots the sequence.
            demo (bool): If True, runs in demo mode.
            standalone (bool): If True, runs the sequence independently.

        Returns:
            bool: Indicates success or failure of the sequence run.
        """
        self.demo = demo  # Set demo mode
        self.plotSeq = plot_seq
        self.standalone = standalone

        '''
        Step 1: Define the interpreter for FloSeq/PSInterpreter.
        The interpreter is responsible for converting the high-level pulse sequence description into low-level
        instructions for the scanner hardware.
        '''

        flo_interpreter = PSInterpreter(
            tx_warmup=hw.blkTime,  # Transmit chain warm-up time (us)
            rf_center=hw.larmorFreq * 1e6,  # Larmor frequency (Hz)
            rf_amp_max=hw.b1Efficiency / (2 * np.pi) * 1e6,  # Maximum RF amplitude (Hz)
            gx_max=hw.gFactor[0] * hw.gammaB,  # Maximum gradient amplitude for X (Hz/m)
            gy_max=hw.gFactor[1] * hw.gammaB,  # Maximum gradient amplitude for Y (Hz/m)
            gz_max=hw.gFactor[2] * hw.gammaB,  # Maximum gradient amplitude for Z (Hz/m)
            grad_max=np.max(hw.gFactor) * hw.gammaB,  # Maximum gradient amplitude (Hz/m)
            grad_t=hw.grad_raster_time * 1e6,  # Gradient raster time (us)
        )

        '''
        Step 2: Define system properties using PyPulseq (pp.Opts).
        These properties define the hardware capabilities of the MRI scanner, such as maximum gradient strengths,
        slew rates, and dead times. They are typically set based on the hardware configuration file (`hw_config`).
        '''

        system = pp.Opts(
            rf_dead_time=hw.blkTime * 1e-6,  # Dead time between RF pulses (s)
            max_grad=np.max(hw.gFactor) * 1e3,  # Maximum gradient strength (mT/m)
            grad_unit='mT/m',  # Units of gradient strength
            max_slew=hw.max_slew_rate,  # Maximum gradient slew rate (mT/m/ms)
            slew_unit='mT/m/ms',  # Units of gradient slew rate
            grad_raster_time=hw.grad_raster_time,  # Gradient raster time (s)
            rise_time=hw.grad_rise_time,  # Gradient rise time (s)
            rf_raster_time=1e-6,
            block_duration_raster=1e-6
        )

        '''
        Step 3: Perform any calculations required for the sequence.
        In this step, students can implement the necessary calculations, such as timing calculations, RF amplitudes, and
        gradient strengths, before defining the sequence blocks.
        '''
        nScans = self.mapVals['nScans']
        params = np.array(np.loadtxt(str(os.path.join(parent_directory_imrdparams, self.mapVals['file']))))
        inversion_time = np.round(params[0],2) * 1e-3
        spokeAxis=self.mapVals['spokeAxis']
        nRepetitions = int((len(params) - 1 ) / 2)
        TRs = np.round(params[1:1 + nRepetitions],2) * 1e-3  # s
        self.mapVals['TRs'] = TRs
        nPoints= self.mapVals['nPoints']

        adc_duration = self.mapVals['acquistionTime'] * 1e-3
        bandwidth_short = (nPoints + 2*hw.addRdPoints) / adc_duration
        sampling_period_short = 1 / bandwidth_short   # us


        '''
        Step 4: Define the experiment to get the true bandwidth
        In this step, student needs to get the real bandwidth used in the experiment. To get this bandwidth, an
        experiment must be defined and the sampling period should be obtained using get_
        '''

        if not demo:
            #Short TR acquisition
            expt = ex.Experiment(
                lo_freq=hw.larmorFreq,  # Larmor frequency in MHz
                rx_t=sampling_period_short*1e6,  # Sampling time in us
                init_gpa=False,  # Whether to initialize GPA board (False for True)
                gpa_fhdo_offset_time=(1 / 0.2 / 3.1),  # GPA offset time calculation
                auto_leds=True  # Automatic control of LEDs (False or True)
            )
            sampling_period_short = expt.get_sampling_period()  # us
            bandwidth_short = 1 / sampling_period_short  # MHz
            print("Acquisition bandwidth fixed to: %0.3f kHz" % (bandwidth_short * 1e3))
            sampling_period_short = sampling_period_short * 1e-6  # s
            expt.__del__()
        else:
            bandwidth_short *= 1e-6  # MHz
            self.mapVals['bw_short_kHz'] = bandwidth_short * 1e3


        #self.mapVals['sampling_period_us'] = sampling_period

        '''
        Step 5: Define sequence blocks.
        In this step, you will define the building blocks of the MRI sequence, including the RF pulses, gradient pulses,
        and ADC blocks.
        '''

        ## Excitation pulse
        # Define the RF excitation pulse using PyPulseq. The flip angle is typically in radians.
        rf_ex_dict = {}  # Crear un diccionario vacío para almacenar los pulsos
        rf_ex_pi_dict={}
        rf_rho_dict={}
        rf_rho_dict[0] = pp.make_block_pulse(flip_angle=np.pi/2,system=system,duration=self.rfExTime,delay=0,phase_offset=0.0)
        rf_rho_dict[1] = pp.make_block_pulse(flip_angle=np.pi,system=system,duration=self.rfExTime,delay=0,phase_offset=0.0)

        rfExFA=params[1+nRepetitions:]
        rf_ex_inversion = pp.make_block_pulse(
            flip_angle=np.pi,  # Set the flip angle for the RF pulse
            system=system,  # Use the system properties defined earlier
            duration=self.rfExTime,  # Set the RF pulse duration
            delay=0,  # Delay before the RF pulse (if any)
            phase_offset=0.0,  # Set the phase offset for the pulse (0 by default)
        )

        nEchos= self.mapVals['nEchos']
        echoSpacing = TRs / nEchos
        jj=0
        for kk in range (nRepetitions):
            flip_ex = rfExFA[kk]  # Convert flip angle from degrees to radians
            rf_ex_dict[kk] = pp.make_block_pulse(
                flip_angle=flip_ex,  # Set the flip angle for the RF pulse
                system=system,  # Use the system properties defined earlier
                duration=self.rfExTime,  # Set the RF pulse duration
                delay=0,  # Delay before the RF pulse (if any)
                phase_offset=0.0,  # Set the phase offset for the pulse (0 by default)
            )
            for echoIndex in range(nEchos):
                if  self.mapVals['piPulseType'] == 'CP':
                    phase = 0.0
                elif  self.mapVals['piPulseType'] == 'CPMG':
                    phase = np.pi / 2
                elif  self.mapVals['piPulseType'] == 'APCP':
                    phase = ((-1) ** (echoIndex) + 1) * np.pi / 2
                elif  self.mapVals['piPulseType'] == 'APCPMG':
                    phase = (-1) ** echoIndex * np.pi / 2

                rf_ex_pi_dict[jj] = pp.make_block_pulse(
                        flip_angle=np.pi,  # Set the flip angle for the RF pulse
                        system=system,  # Use the system properties defined earlier
                        duration=self.rfExTime,  # Set the RF pulse duration
                        delay=0,  # Delay before the RF pulse (if any)
                        phase_offset= phase,  # Set the phase offset for the pulse (0 by default)
                )
                jj+=1


        ## ADC block
        # Define the ADC block using PyPulseq. You need to specify number of samples and delay.
        adc_dict_short={}
        adc_rho_dict={}
        adc_rho_dict[0]=pp.make_adc(num_samples= nPoints + 2*hw.addRdPoints,dwell=sampling_period_short,delay= self.rfExTime + hw.deadTime*1e-6)

        adc_index=0
        for kk in range (0,nRepetitions):
            for ll in range (nEchos):
                acqpoints = nPoints + 2*hw.addRdPoints
                blk = hw.blkTime
                ddt = hw.deadTime
                tau = TRs[kk] / nEchos
                tau = round(tau / hw.grad_raster_time) * hw.grad_raster_time
                delay_acq = half_tau = round((tau / 2) / hw.grad_raster_time) * hw.grad_raster_time - adc_duration / 2
                adc_dict_short[adc_index] = pp.make_adc(
                    num_samples= acqpoints,
                    dwell=sampling_period_short,
                    delay= delay_acq
                )
                adc_index+=1





        ## GRADIENT FOR 1 SPOKE
        grad_dict={}
        ll=0

        if self.mapVals['spoke']==1:
            gradamp = 10  # mT/m, ajustar si hace falta
        else:
            gradamp = 0
        # Gradiente de subida inicial (rampa)
        grad_dict[ll] = pp.make_extended_trapezoid(
            spokeAxis,
            amplitudes=[0, gradamp],
            times=[0, hw.grad_rise_time],
            max_slew=hw.max_slew_rate,
            system=system
        )
        ll+=1
        # Gradientes constantes durante cada repetición
        for kk in range(nRepetitions):
            # Duración de cada segmento entre ecos
            tau = TRs[kk] / nEchos

            # Redondear tau a múltiplo del grad_raster_time
            tau = round(tau / hw.grad_raster_time) * hw.grad_raster_time
            half_tau = round((tau / 2) / hw.grad_raster_time) * hw.grad_raster_time
            grad_dict[ll] = pp.make_extended_trapezoid(
                spokeAxis,
                amplitudes=[gradamp,gradamp],
                times=[0,half_tau],
                max_slew=hw.max_slew_rate,
                system=system
            )
            ll+=1
            for jj in range (nEchos-1):
                grad_dict[ll] = pp.make_extended_trapezoid(
                    spokeAxis,
                    amplitudes=[gradamp, gradamp],
                    times=[0, tau],
                    max_slew=hw.max_slew_rate,
                    system=system
                )
                ll += 1
            grad_dict[ll] = pp.make_extended_trapezoid(
                spokeAxis,
                amplitudes=[gradamp,gradamp],
                times=[0,half_tau],
                max_slew=hw.max_slew_rate,
                system=system
            )
            ll+=1


        # Gradiente de bajada al final
        grad_dict[ll] = pp.make_extended_trapezoid(
            spokeAxis,
            amplitudes=[gradamp, 0],
            times=[0, hw.grad_rise_time],
            max_slew=hw.max_slew_rate,
            system=system
        )

        ## Repetition delay
        # Define the delay for repetition.

        # Additional considerations for students:
        # - Make sure timing calculations account for hardware limitations, such as gradient raster time and dead time.

        '''
        Step 6: Define your initializeBatch according to your sequence.
        In this step, you will create the initializeBatch method to create dummy pulses that will be initialized for
        each new batch.
        '''

        def initializeBatch():
            """
            Initializes a batch of MRI sequence blocks using PyPulseq for a given experimental configuration.

            Returns:
            --------
            tuple
                - batch (pp.Sequence): A PyPulseq sequence object containing the configured sequence blocks.
                - n_rd_points (int): Total number of readout points in the batch.
                - n_adc (int): Total number of ADC acquisitions in the batch.
            """

            # Instantiate pypulseq sequence object
            batch = pp.Sequence(system)
            n_rd_points = 0
            n_adc = 0
            return batch, n_rd_points, n_adc


        '''
        Step 7: Define your createBatches method.
        In this step you will populate the batches adding the blocks previously defined in step 4, and accounting for
        number of acquired points to check if a new batch is required.
        '''
        def createBatches(case='short'):
            """
            Create batches for the full pulse sequence.

            Instructions:
            - This function creates the complete pulse sequence by iterating through repetitions.
            - Each iteration adds new blocks to the sequence, including the RF pulse, ADC block, and repetition delay.
            - If a batch exceeds the maximum number of readout points, a new batch is started.

            Returns:
                waveforms (dict): Contains the waveforms for each batch.
                n_rd_points_dict (dict): Dictionary of readout points per batch.
                n_adc (int): Total number of ADC acquisitions across all batches
            """
            batches = {}  # Dictionary to save batches PyPulseq sequences
            waveforms = {}  # Dictionary to store generated waveforms per each batch
            n_rd_points_dict = {}  # Dictionary to track readout points for each batch
            n_rd_points = 0  # To account for number of acquired rd points
            seq_idx = 0  # Sequence batch index
            n_adc = 0  # To account for number of adc windows
            batch_num = "batch_0"  # Initial batch name

            nRepetitions = int((len(params) - 1) / 2)
            # Loop through all repetitions (e.g., slices)
            for repetition in range(nScans):
                # Check if a new batch is needed (either first batch or exceeding readout points limit)
                if seq_idx == 0 or n_rd_points + self.nPoints > hw.maxRdPoints:
                    # If a previous batch exists, write and interpret it
                    if seq_idx > 0:
                        batches[batch_num].write(batch_num + ".seq")
                        waveforms[batch_num], param_dict = flo_interpreter.interpret(batch_num + ".seq")
                        print(f"{batch_num}.seq ready!")

                    # Update to the next batch
                    seq_idx += 1
                    n_rd_points_dict[batch_num] = n_rd_points  # Save readout points count
                    n_rd_points = 0
                    batch_num = f"batch_{seq_idx}_{case}"
                    batches[batch_num], n_rd_points, n_adc_0 = initializeBatch()  # Initialize new batch
                    n_adc += n_adc_0
                    print(f"Creating {batch_num}.seq...")

                # Add sequence blocks (RF, ADC, repetition delay) to the current batch
                long_position=np.zeros(0)

                batches[batch_num].add_block(pp.make_delay(hw.deadTime * 1e-6))
                batches[batch_num].add_block(pp.make_adc(num_samples= nPoints + 2*hw.addRdPoints,dwell=sampling_period_short,delay= 0),pp.make_extended_trapezoid(spokeAxis, amplitudes=[0, 0],times=[0, 2*adc_duration],max_slew=hw.max_slew_rate,system=system))  #Medida de ruido
                n_rd_points += self.nPoints + 2 * hw.addRdPoints
                n_adc+=1
                batches[batch_num].add_block(rf_rho_dict[0])
                batches[batch_num].add_block(pp.make_delay(hw.deadTime*1e-6))
                batches[batch_num].add_block(rf_rho_dict[1],adc_rho_dict[0],pp.make_extended_trapezoid(spokeAxis, amplitudes=[0, 0],times=[0, 2*adc_duration],max_slew=hw.max_slew_rate,system=system))
                batches[batch_num].add_block(pp.make_delay(10))
                n_rd_points+= self.nPoints + 2 *hw.addRdPoints
                n_adc += 1

                batches[batch_num].add_block(rf_ex_inversion)  # Inversion
                batches[batch_num].add_block(pp.make_delay(
                    inversion_time - batches[batch_num].block_durations[list(batches[batch_num].block_durations)[-1]]-hw.grad_rise_time))
                grad_index = 0
                batches[batch_num].add_block(grad_dict[grad_index])  # Grad rise
                grad_index+=1
                if case == 'short':
                    ll=0
                    for kk in range(nRepetitions):
                        batches[batch_num].add_block(rf_ex_dict[kk], grad_dict[grad_index])
                        grad_index += 1
                        for echos in range(nEchos-1):
                            batches[batch_num].add_block(rf_ex_pi_dict[nEchos*kk +echos ], adc_dict_short[nEchos*kk +echos],grad_dict[grad_index])
                            grad_index += 1
                            n_rd_points += int(self.nPoints) + 2 * hw.addRdPoints # Accounts for additional acquired points in each adc block
                            n_adc += 1
                        batches[batch_num].add_block(rf_ex_pi_dict[nEchos*kk + nEchos-1],grad_dict[grad_index])
                        grad_index+=1

                    batches[batch_num].add_block(grad_dict[2 * nRepetitions + 1])  # Grad down
                    batches[batch_num].add_block(pp.make_delay(10))
            # After final repetition, save and interpret the last batch
            batches[batch_num].write(batch_num + ".seq")
            waveforms[batch_num], param_dict = flo_interpreter.interpret(batch_num + ".seq")
            print(f"{batch_num}.seq ready!")
            print(f"{len(batches)} batches created. Sequence ready!")

            # Update the number of acquired ponits in the last batch
            n_rd_points_dict.pop('batch_0')
            n_rd_points_dict[batch_num] = n_rd_points
            self.mapVals['long_position'] = long_position

            return waveforms, n_rd_points_dict, n_adc

        '''
        Step 8: Run the batches
        This step will handle the different batches, run it and get the resulting data. This should not be modified.
        Oversampled data will be available in self.mapVals['data_over']
        Decimated data will be available in self.mapVals['data_decimated']
        The decimated data is shifted to account for CIC delay, so data is synchronized with real-time signal
        '''

        waveforms_short, n_readouts_short, n_adc_short = createBatches(case='short')

        # Run sequence a
        self.runBatches(waveforms=waveforms_short,
                               n_readouts=n_readouts_short,
                               n_adc=n_adc_short,
                               frequency=hw.larmorFreq,  # MHz
                               bandwidth=bandwidth_short,  # MHz
                               decimate='Normal',
                               hardware=True,
                               output='short'
                               )
        return True


    def sequenceAnalysis(self, mode=None):

        data_short = self.mapVals['data_decimated_short']
        nPoints = self.mapVals['nPoints']
        nTRs = len(self.mapVals['TRs'])
        nEchos = self.mapVals['nEchos']
        data_concatenated = np.zeros(0)

        noisemeasure= data_short[hw.addRdPoints : nPoints + hw.addRdPoints]
        data_short = data_short[nPoints + 2 * hw.addRdPoints:]
        rhopoints=data_short[hw.addRdPoints : nPoints + hw.addRdPoints]
        data_short=data_short[nPoints+2*hw.addRdPoints :]
        for kk in range (nTRs*(nEchos-1)):
            data_concatenated = np.concatenate((data_concatenated,data_short[hw.addRdPoints:hw.addRdPoints + nPoints ]), axis=0 )
            data_short = data_short[nPoints+ 2 * hw.addRdPoints :]

        result1 = {'widget': 'curve',
                   'xData': np.linspace(0,nTRs,nTRs*(nEchos-1)*nPoints),
                   'yData': [np.real(data_concatenated), np.imag(data_concatenated)],
                   'xLabel': 'Time points (1/TR)',
                   'yLabel': 'Signal amplitude (mV)',
                   'title': 'Evolution during signal',
                   'legend': ['real', 'imag'],
                   'row': 0,
                   'col': 0}

        # create self.out to run in iterative mode
        self.output = [result1]

        # save data once self.output is created
        self.saveRawData()

        # Plot result in standalone execution
        if self.mode == 'Standalone':
            self.plotResults()

        return self.output

if __name__=="__main__":
    seq = IMRD()
    seq.sequenceAtributes()
    seq.sequenceRun(plot_seq=False, demo=True, standalone=True)
    # seq.sequenceAnalysis(mode='Standalone')
    