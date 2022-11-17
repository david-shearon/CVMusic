from midiutil.MidiFile import MIDIFile


# take in a array of note strings with value and fractional value 
# (e.g. [['C4', 1], ['E4', 2]]) would represent middle C, a 1/1 (whole) note, and the E just above that, a 1/2 (half) note
# and turn them into a midi file, playing back right after
def play_music(note_arr, output_name, tempo):
    midi_compiler = MIDIFile(1)
    midi_compiler.addTempo(0, 0, tempo)     # Will be saved at tempo bpm
    midi_compiler.addTrackName(0, 0, output_name)

    # for each note in the array, turn note string into a midi note object
    # will need to translate the note value into a frequency - use a mapping
    note_frequency_map = {
         'C0': 16.35,
         'D0': 18.35,
         'E0': 20.60,
         'F0': 21.83,
         'G0': 24.50,
         'A0': 27.50,
         'B0': 30.87,
         'C1': 32.70,
         'D1': 36.71,
         'E1': 41.20,
         'F1': 43.65,
         'G1': 49.00,
         'A1': 55.00,
         'B1': 61.74,
         'C2': 65.41,
         'D2': 73.42,
         'E2': 82.41,
         'F2': 87.31,
         'G2': 98.00,
         'A2': 110.00,
         'B2': 123.47,
         'C3': 130.81,
         'D3': 146.83,
         'E3': 164.81,
         'F3': 174.61,
         'G3': 196.00,
         'A3': 220.00,
         'B3': 246.94,
         'C4': 261.63,
         'D4': 293.66,
         'E4': 329.63,
         'F4': 349.23,
         'G4': 392.00,
         'A4': 440.00,
         'B4': 493.88,
         'C5': 523.25,
         'D5': 587.33,
         'E5': 659.25,
         'F5': 698.46,
         'G5': 783.99,
         'A5': 880.00,
         'B5': 987.77,
         'C6': 1046.50,
         'D6': 1174.66,
         'E6': 1318.51,
         'F6': 1396.91,
         'G6': 1567.98,
         'A6': 1760.00,
         'B6': 1975.53,
         'C7': 2093.00,
         'D7': 2349.32,
         'E7': 2637.02,
         'F7': 2793.83,
         'G7': 3135.96,
         'A7': 3520.00,
         'B7': 3951.07,
         'C8': 4186.01,
         'D8': 4698.63,
         'E8': 5274.04,
         'F8': 5587.65,
         'G8': 6271.93,
         'A8': 7040.00,
         'B8': 7902.13,
    }


    for note_idx in range(len(note_arr)):
        note_pair = note_arr[note_idx]
        note_value = note_pair[0]
        note_duration = 4 * (1 / note_pair[1])  # 4 beats in a whole note
        freq_value = note_frequency_map[note_value]
        midi_compiler.addNote(0, 0, note_value, note_idx, note_duration, volume=100)

    # ready to write to file
    output_stream = open(output_name + ".mid", 'w')
    midi_compiler.writeFile(output_stream)

        