from midiutil.MidiFile import MIDIFile


# take in a array of note strings with value and fractional value 
# (e.g. [['C4', 1], ['E4', 2]]) would represent middle C, a 1/1 (whole) note, and the E just above that, a 1/2 (half) note
# and turn them into a midi file, playing back right after
def save_music(note_arr, output_name, tempo):
    midi_compiler = MIDIFile(1)
    midi_compiler.addTempo(0, 0, tempo)     # Will be saved at tempo bpm
    midi_compiler.addTrackName(0, 0, output_name)

    # for each note in the array, turn note string into a midi note object
    # will need to translate the note value into a tone - use a mapping
    # C4 starts at pitch 60, and adjacent notes increase pitch by 1. We aren't worried about sharps/flats
    note_pitch_map = {
          'C': 60,
          'D': 62,
          'E': 64,
          'F': 65,
          'G': 67,
          'A': 69,
          'B': 71
    }
    song_beat = 0
    for note_idx in range(len(note_arr)):
        note_pair = note_arr[note_idx]
        note_letter = note_pair[0][0]   # ex: 'C'
        if note_letter == 'R':
            # rest - don't write a beat for as long as this rest lasts for
            note_duration = int(4 * (1 / note_pair[1]))  # 4 beats in a whole rest
            song_beat += note_duration  # next note must start on beat that this rest finished on
        else:
            note_octave = int(note_pair[0][1])   # ex: 4
            # using octave, compare to the 4th octave notes defined in the map
            tone_value = note_pitch_map[note_letter] + (12 * (note_octave - 4))
            note_duration = int(4 * (1 / note_pair[1]))  # 4 beats in a whole note
            # add current note to midi object
            midi_compiler.addNote(0, 0, tone_value, song_beat, note_duration, volume=100)
            song_beat += note_duration  # next note must start on beat that the last note finished on

    # ready to write to file
    with open(output_name + ".mid", 'wb') as file_handler:
        midi_compiler.writeFile(file_handler)


