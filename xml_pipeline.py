
import xml.etree.ElementTree as ET
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector
from qiskit_aer.noise import NoiseModel, phase_damping_error
from qiskit_aer import AerSimulator
import qmuvi


# Next Steps
        # (Week One) Complete Quantum Circuit representation of XML file
        # (Week Two) Successful sonify the quantum circuit
        # (Week Three) Introduce Probabilistic Elements
        # (Week Four) Begin compiling demonstration for SPIN L.T.

# Notes about quantum representation
        # One big change I incorporated over the midi to quantum pipeline is the definition of a function
            # that explicitly defines musical continuity between frames. This is defined as
            # the union between the set of currently active and previously active notes 
            # across frames.
        # When building the actual quantum circuit, we use this value of note['continued']
            # to determine how we manage each individual wave function. If the wave function
            # is observing a note that is continued, the coefficients, which you can think
            # of as the probability associated with each pitch in the frame, are carried over.
            # You can think of this as evolving the state rather than reinitalizing it.
            # otherwise, the wave function is reinitialized with the frame's pitch values and 
            # normalized.


# Improvements I have noticed over MIDI-pipeline
        # Frame generation is much more articulated, we are able to carry over continued
        # notes across frames rather than "re-triggering notes" when we measure each wavefunction

# Incorporating melody, pitch, harmony and rhythm, divisions, articulation and timbre, 
# get probabilities from vast selection of given composer's pieces
# associated probability for every next note, large dataset of note -> note -> note values

# 3 maroon 5 songs, look at melody, create data of note succession probabilities. Probability
# from going from one note to the next in the fourth song
# Takes in a musicXML file and returns a structured representation of musical data, ie. (notes, measures, durations)
# Output structure
# score = {
#     "parts": {
#         "P1": {
#             "measures": [
#                 {
#                     "number": 1,
#                     "notes": [
#                         {
#                             "pitch": 45,       
#                             "duration": 4,
#                             "voice": 1,
#                             "is_rest": False
#                         }
#                     ]
#                 }
#             ]
#         }
#     }
# }
def parse_musicxml(root):
    
    STEP_TO_SEMITONE = {
    "C": 0, "D": 2, "E": 4,
    "F": 5, "G": 7, "A": 9, "B": 11
}
    
    score = {"parts" : {}}
    
    for part in root.findall("part"):
        part_id = part.attrib["id"]
        
        score["parts"][part_id] = {"measures": []}
        measurements = score["parts"][part_id]["measures"]
        prev_attributes = None
        prev_tempo = None
        for measure in part.findall("measure"):            
            attrs = measure.find("attributes")
            if attrs is not None:
                key = attrs.find("key")
                time = attrs.find("time")
                tempo = measure.find("sound").attrib["tempo"] if measure.find("sound") is not None else prev_tempo
                attributes = {
                    "division": int(attrs.findtext("divisions", default="0")),
                    "key" : {
                        "fifths": int(key.findtext("fifths", default=0)) if key is not None else 0,
                        "mode": key.findtext("mode", "major") if key is not None else "major"
                    },
                    "time" : {
                        "beats" : int(time.findtext("beats", default=4)) if time is not None else 4,
                        "beat_type" : int(time.findtext("beat-type", default=4)) if time is not None else 4
                    }
                }
                prev_attributes = attributes
                prev_tempo = tempo
            else:
                attributes = prev_attributes
                tempo = prev_tempo
                
            number = measure.attrib["number"]

            notes = []
            for note in measure.findall("note"):
                duration = int(note.findtext("duration", default="0"))
                is_rest = note.find("rest") is not None
                voice = int(note.findtext("voice", default="1"))
                
                midi_pitch_value = None
                if not is_rest:
                    pitch = note.find("pitch")
                    step = pitch.findtext("step")
                    octave = int(pitch.findtext("octave"))
                    alter = int(pitch.findtext("alter", default="0"))
                    midi_pitch_value = 12 * (octave + 1) + STEP_TO_SEMITONE[step] + alter
                notes.append({
                    "pitch" : midi_pitch_value,
                    "duration" : duration,
                    "voice" : voice,
                    "is_rest" : is_rest
                })
            measure_dict = {
                "number": number,
                "notes": notes,
                "attributes": attributes,
                "tempo": tempo
            }
            measurements.append(measure_dict)
    return score

# Takes in score to get global timing and structural elements, ie. (tempo)

# global_features = {
#     "divisions": 2,
#     "tempo": [
#         {"measure": 1, "bpm": 120}
#     ],
#     "time_signature": {
#         "beats": 4,
#         "beat_type": 4
#     },
#     "key_signature": {
#         "fifths": 0,
#         "mode": "major"
#     }
# }

def get_global_features(score):
    global_features = {
        "division": None,
        "tempo": [],
        "time_signature": [],
        "key_signature": []
    }
    prev_tempo = None
    prev_time = None
    prev_key = None
    # Get Division number (time units that make up one quarter note)
    global_features['division'] = score['parts']['P1']['measures'][0]['attributes']['division']
    
    # For each measure extract the global features that are different from prior measure
    for part in score['parts'].values():
        for measure in part['measures']:
            number = measure['number']
            attributes = measure['attributes']
            if attributes is not None:
                division = attributes["division"]
                if global_features["division"] is None:
                    global_features["division"] = division
            
            tempo = measure['tempo']
            if tempo != prev_tempo:
                tempo_dict = {'measure' : number, 'bpm' : tempo}
                prev_tempo = tempo
                global_features['tempo'].append(tempo_dict)

            time_sig = attributes["time"]
            if time_sig != prev_time:
                global_features["time_signature"].append({
                    "measure": number,
                    **time_sig
                })
                prev_time = time_sig

            key_sig = attributes["key"]
            if key_sig != prev_key:
                global_features["key_signature"].append({
                    "measure": number,
                    **key_sig
                })
                prev_key = key_sig
                
    # Set Default Tempo
    if len(global_features['tempo']) == 0:
        global_features['tempo'].append({'measure' : 1, 'bpm' : 120})
    return global_features

def extract_instruments(root):
    """
    Extract instrument metadata from a MusicXML <part-list>.

    Returns:
        dict mapping part_id -> instrument info
    """
    instruments = {}

    part_list = root.find("part-list")
    if part_list is None:
        return instruments

    for score_part in part_list.findall("score-part"):
        part_id = score_part.attrib["id"]

        part_name = score_part.findtext("part-name", default="Unknown")

        # Default values
        instrument_name = part_name
        midi_program = None
        midi_channel = None

        score_instr = score_part.find("score-instrument")
        if score_instr is not None:
            instrument_name = score_instr.findtext(
                "instrument-name", default=part_name
            )

        midi_instr = score_part.find("midi-instrument")
        if midi_instr is not None:
            midi_program = int(midi_instr.findtext("midi-program", default="0"))
            midi_channel = int(midi_instr.findtext("midi-channel", default="1"))

        instruments[part_id] = {
            "part_name": part_name,
            "instrument_name": instrument_name,
            "midi_program": midi_program,
            "midi_channel": midi_channel
        }

    return instruments

        

# creates time-ordered musical events that are easily rendered by a quantum computer
# an event denotes one musical note occurrence
# Includes when a specific pitch begins, how long it lasts, and which voice it belongs to
def create_events(score, global_features):
    events = []

    division = global_features["division"]

    for part_id, part in score["parts"].items():

        # track time per voice
        voice_time = {}

        for measure in part["measures"]:
            for note in measure["notes"]:
                voice = note["voice"]
                duration_beats = note["duration"] / division

                # initialize voice cursor
                if voice not in voice_time:
                    voice_time[voice] = 0.0

                start_time = voice_time[voice]

                if not note["is_rest"]:
                    events.append({
                        "time": start_time,
                        "pitch": note["pitch"],
                        "duration": duration_beats,
                        "part": part_id,
                        "voice": voice
                    })

                # advance voice cursor
                voice_time[voice] += duration_beats

    # sort all events globally by time
    events.sort(key=lambda e: e["time"])

    return events

# Creates a list of frames that divides the music into time intervals, where the set 
# of audible notes remains unchanged
def build_frames(events):
    
    # Collect all note on/off boundaries
    boundaries = set()

    for e in events:
        boundaries.add(e["time"])
        boundaries.add(e["time"] + e["duration"])

    # Sort time boundaries
    times = sorted(boundaries)

    frames = []

    for i in range(len(times) - 1):
        t_start = times[i]
        t_end = times[i + 1]
        frame_duration = t_end - t_start

        if frame_duration <= 0:
            continue

        active_notes = []

        for e in events:
            if e["time"] <= t_start < e["time"] + e["duration"]:
                active_notes.append({
                    "pitch": e["pitch"],
                    "part": e["part"],
                    "voice": e["voice"]
                })

        if active_notes:
            frames.append({
                "start": t_start,
                "duration": frame_duration,
                "notes": active_notes
            })

    return frames


# One big change I incorporated over the midi to quantum pipeline is the definition of a function
            # that explicitly defines musical continuity between frames. This is defined as
            # the union between the set of currently active and previously active notes 
            # across frames.
            
def apply_musical_elements(frames):
    # creates set containing active note values
    prev_active = set()
    
    # for each frame, use set operations to determine whether or not the note is continued, new, or ending
    for frame in frames:
        curr_active = set((n['pitch'], n['part'], n['voice']) for n in frame['notes'])
        frame['continued'] = curr_active & prev_active
    
        # By defining whether or not a note is continued
        # we can prevent a wave function from reinitialzing the values
        for note in frame['notes']:
            note['continued'] = (note['pitch'], note['part'], note['voice']) in frame['continued']
        frame['new_note'] = curr_active - prev_active
        frame['end_note'] = prev_active - curr_active
        prev_active = curr_active
    return frames


# musical frames are transformed into wavefunctions
# pitch -> basis state 
# chord -> superposition
# Duration -> amplitude

        # When building the actual quantum circuit, we use the value of note['continued']
            # (derived from the apply_musical_elements function)
            # to determine how we manage each individual wave function. If the wave function
            # is observing a note that is continued, the coefficients, which you can think
            # of as the sqrt(probability) associated with each pitch in the frame, are carried over.
            # You can think of this as evolving the state rather than reinitalizing it.
            # otherwise, the wave function is reinitialized with the frame's pitch values and 
            # normalized.
            
def build_wave_functions(frames, dim=128):
    n_frames = len(frames)
    wave_functions = []
    for i in range(n_frames):
        wave_functions.append(np.zeros(dim, dtype=complex))
    for idx, frame in enumerate(frames):
        for note in frame['notes']:
            pitch = note['pitch']
            # if the note is continued, copy the pitches from the previous wave function. 
            # this preserves note identity across frames
            # prevents the re-triggering of notes we heard in the MIDI-quantum pipeline
            if note.get("continued", False):
                wave_functions[idx][pitch] = wave_functions[idx - 1][pitch]
            else:
                wave_functions[idx][pitch] = 1
        if np.linalg.norm(wave_functions[idx]) > 0:
            wave_functions[idx] /= np.linalg.norm(wave_functions[idx])
    return wave_functions

def build_circuit(wave_functions):
    n_qubits = int(np.log2(len(wave_functions[0])))
    qc = QuantumCircuit(n_qubits)
    for psi in wave_functions:
        qc.initialize(psi, range(n_qubits))
        qc.barrier()
    return qc

# My idea for first quantum phenomena: Quantum Entanglement

    # Let's entangle pitches and instruments
    
        # Classically: Introduce a deterministic rule that assigns an instrument to a known pitch
            # If pitch is low -> violin
            # If pitch is high -> piano
            # pitch = f(instrument)
            
            # Test:
                # Run the code and check pitch mappings. 
                # Change instrument mapping
                # Pitch should remain unchanged
                # So changes to instrument should not affect pitch behavior
            
        # Quantum:
            # pitch = lower 6 bits of quantum circuit. i & 0b00111111
            # instrument = top bit (i>>6) & 0b1
            # map pitch  value to instrument (piano or voilin)
            
            # Test:
                # Entangle (H + cx gate) pitch and instrument. This means the two only exist in their joint quantum state. Always correlated
                # Rotate instrument qubits, notice that instrument also changes.
                
                
# QMuVi sonifies pure states, not qubits; instruments are chosen by phase, not by index bits—so timbre must be expressed through phase–amplitude correlations, not through bit routing.

tree = ET.parse("xml_files/chariotsoffire.musicxml")
root = tree.getroot()
score = parse_musicxml(root)
global_features = get_global_features(score)
instruments = extract_instruments(root)
instrument_arr = []
for part in instruments.values():
    instrument_arr.append(part['midi_program'] - 1)
events = create_events(score, global_features)

frames = build_frames(events)
frames = apply_musical_elements(frames)
wave_functions = build_wave_functions(frames)
qc = build_circuit(wave_functions)

# Toying around with mixed states

# >>>>>>>
# psi1 = wave_functions[0]
# psi2 = wave_functions[30]

# rho1 = DensityMatrix(Statevector(psi1))
# rho2 = DensityMatrix(Statevector(psi2))

# p = 0.6
# rho = p * rho1 + (1-p) * rho2

# purity = np.trace(rho.data @ rho.data).real
# print(f"Purity: {purity}")
# >>>>>>>>>

QMUV_TPQ = 480  # QMuVi ticks per quarter note


def note_map(i: int) -> int:
    return int(i)

rhythm = []
for frame in frames:
    duration_beats = frame["duration"]
    ticks = max(60, int(duration_beats * QMUV_TPQ))
    rhythm.append([ticks, 0, 10])


# All qmuvi is doing is taking the indices of the wavefunction with non-zero probability values
#   (each index corresponds to a midi note) and measuring them when 
#   we reach a barrier in the quantum circuit. So as long as we introduce our
#   quantum gates prior to feeding it into qmuvi, we are not limited in our ability to
#   experiment with different quantum behavior.
qmuvi.generate_qmuvi(
    qc,
    "chariots_of_fire",
    note_map = note_map,
    instruments=instrument_arr,
    rhythm = rhythm,
    fps = 24,
    smooth_transitions = True
)