using DataFrames
using Requests

BIRDDB_FILE_BASE_URL = "http://taylor0.biology.ucla.edu/birdDBQuery/Files/"
RECORDINGS_DIR = "recordings/"
TEXTGRID_DIR = "textgrid_files/"

CSV_INPUT_FILE = "BirdDB Query Results.csv"
CSV_WITH_FILENAMES = "birddb.csv"

# Define main func so that all the helper functions can be
# defined together later in the file
function main()

    # create the file directories if they don't already exist
    info("Checking if file directories exist already")

    if ! isdir(RECORDINGS_DIR)
        mkdir(RECORDINGS_DIR)
    end

    if ! isdir(TEXTGRID_DIR)
        mkdir(TEXTGRID_DIR)
    end

    # import the csv from the database
    info("Reading csv of database dump")

    birddb_data = readtable(CSV_INPUT_FILE)
    # use less data for testing
    #birddb_data = birddb_data[1:5,:]

    birddb_data[:recording_file_url] = get_recording_filepaths(birddb_data[:Textgrid_file])

    birddb_data[:recording_filename] = map( r -> "$(RECORDINGS_DIR)$(replace(r, r"(/)", s"-"))", birddb_data[:recording_file_url])
    birddb_data[:textgrid_filename] = map( r -> "$(TEXTGRID_DIR)$(replace(r, r"(/)", s"-"))", birddb_data[:Textgrid_file])

    info("Starting to download the files")
    download_files(birddb_data)

    info("Writing out a csv with the filenames attached to $CSV_WITH_FILENAMES")
    writetable(CSV_WITH_FILENAMES, birddb_data)
end

function get_recording_filepaths(textgrid_filepathes::DataArrays.DataArray{UTF8String,1})
    # because the CSV only gives you the actual text and not the hyperlink
    # it doesn't contain the correct info to pull the recording file from
    # the server
    # but the textgrid path happens to almost match the recording path so
    # we'll use that instead

    recording_array = Array{UTF8String, 1}()

    # this could probably be parallel but I wasn't sure if order was gauranteed when
    # reducing with vcat
    for t = textgrid_filepathes
        recording_match = match(r"^Files_TextGrids/(\d{4}/\w+/[\w-_]+)\.TextGrid$", t)
        if recording_match == nothing
            warn("Regex match on textgrid filepath failed, searched path: $t")
        end
        recording_path = recording_match.captures[1]

        push!(recording_array, "Tracks/$recording_path.WAV")
    end

    return recording_array
end

function download_files(birddb_data)

    # Download the files and save them to the proper place
    for (url, out_file) in zip(birddb_data[:recording_file_url], birddb_data[:recording_filename])
        info(out_file)
        recording_file = Requests.get("$(BIRDDB_FILE_BASE_URL)$(url)")
        save(recording_file, out_file)
    end

    for (url, out_file) in zip(birddb_data[:Textgrid_file], birddb_data[:textgrid_filename])
        info(out_file)
        textgrid_file = Requests.get("$(BIRDDB_FILE_BASE_URL)$(url)")
        save(textgrid_file, out_file)
    end

    nothing
end

main()

