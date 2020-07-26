# This section will have all the inputs
import base64
import pickle
import copy
from datetime import datetime
from io import BytesIO
from typing import Tuple

import datetime
import pandas as pd
import streamlit as st
from PIL import Image
from pandas import ExcelWriter
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN


# reusable functions

# Function to get encodings
def get_encodings(model, sentences):
    encodings = model.encode(sentences)
    return encodings


# Function to get month and year from a date
def get_month(date):
    mth_yr = date.strftime('%b-%Y')
    return mth_yr


# Function to extract a string based on a pattern
def extract_string(input_row, start, end):
    input_string = input_row[6]
    if len(input_string.split(start)) > 1:
        return (input_string.split(start)[1]).split(end)[0]
    else:
        return 'no details'


# Function to extract text after a particular sub-string
def find_substring(input_text, substring):
    start = input_text.find(substring)
    if start == -1:
        return input_text  # Not Found
    else:
        return input_text[start + len(substring):]


# Function to load pickle file - in this case model weights
def load_pkl(obj_to_be_loaded):
    infile = open(obj_to_be_loaded, 'rb')
    output_obj = pickle.load(infile)
    infile.close()
    return output_obj


def to_excel(df):
    output = BytesIO()
    writer: ExcelWriter = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1', float_format="%.2f")
    writer.save()
    processed_data = output.getvalue()
    return processed_data


def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    file_name = f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="Your_File.xlsx">Download As Excel file</a>'  # decode b'abc' => abc
    return file_name


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# This function will cache the weights to avoid running everytime
@st.cache(suppress_st_warning=False, allow_output_mutation=True)
def get_weights():
    weights = SentenceTransformer(r'.\model\roberta-large-nli-mean-tokens')
    return weights


def load_page_head_with_image() -> bool:
    st.header("Incident Analysis Platform based on Natural Language Processing")
    local_css("style.css")
    image = Image.open(r'.\images\CogLogo.PNG')
    st.image(image, caption=None, width=50, format='PNG', use_column_width=True)
    st.text("The input file is expected to have 20 Columns incuding 1.Task type, 2.Assignment group")
    st.text("3.Number, 4.State, 5.Priority, 6.Opened, 7.Opened by, 8.Assigned to, 9.Short Description")
    st.text("10.Closed, 11.Application Criticality, 12.Closed by, 13.Active, 14.CI Name, 15.Updated by")
    st.text("16.SLA Due Date, 17.Work Start, 18.Created, 19.Work notes, 20.Description")
    st.write('-' * 15)
    load_header = True
    return load_header


def load_side_bar_with_options() -> Tuple[bool, float, any, float, any]:
    st.sidebar.markdown("# Select the parameters for Run")
    # Set % mismatch allowed. Set this separately for incidents and others
    # This means we want the tickets to be matched 80%. Need to adjust as needed
    eps_inc = st.sidebar.radio("Percentage Match expected on Incidents?", ('80%', '70%'))

    if eps_inc == '70%':
        eps_inc = 0.30
    else:
        eps_inc = 0.20

    # Minimum number of samples needed for creating a new group.
    min_samples_inc = st.sidebar.radio("Minimum number of samples needed for Incidents?", (3, 2))

    # This means we want the tickets to be matched 80%. Need to adjust as needed
    eps_task = st.sidebar.radio("Percentage Match expected on Tasks?", ('80%', '70%'))

    if eps_task == '70%':
        eps_task = 0.30
    else:
        eps_task = 0.20

    # Minimum number of samples needed for creating a new group.
    min_samples_task = st.sidebar.radio("Minimum number of samples needed for tasks?", (3, 2))

    load_side_bar = True

    return load_side_bar, eps_inc, min_samples_inc, eps_task, min_samples_task


def upload_file_for_processing() -> Tuple[any, bool]:
    exception = False
    upload_file = ''
    try:
        upload_file = st.file_uploader('File Uploader', type='xlsx')
        return upload_file, exception
    except Exception as exception:
        return upload_file, exception


def read_input_file(upload_file) -> Tuple[pd.DataFrame, bool]:
    input_data_df = pd.read_excel(upload_file)
    input_read_f = True
    return input_data_df, input_read_f


def filter_based_on_date(input_df) -> Tuple[pd.DataFrame, bool]:
    # Pick only tickets which are on or after 01-01-2020
    cut_off = datetime.datetime(2020, 1, 1)
    input_df = input_df[input_df['Created'] >= cut_off]
    input_df.reset_index(inplace=True, drop=True)
    date_filter_done_f = True
    return input_df, date_filter_done_f


def pre_processing_step_one(input_df) -> Tuple[pd.DataFrame, bool]:
    input_df = input_df

    # This is used in step#1 and other two is used in step#2
    output_cols3 = ['Task type', 'Number', 'Description', 'Assignment group',
                    'State', 'Priority', 'Work notes', 'Opened', 'Opened by',
                    'Assigned to', 'Closed', 'Application Criticality', 'Closed by', 'SLA Due Date']

    # If Description is empty populate Short Description into Description
    input_df.loc[input_df['Description'].isna(), 'Description'] = input_df.loc[
        input_df['Description'].isna(), 'Short Description']

    # Remove all numbers from Description column
    input_df['Description'] = input_df['Description'].str.replace(r'\d+', '')

    # Remove all numbers from Work notes column
    input_df['Work notes'] = input_df['Work notes'].str.replace(r'\d+', '')

    # Only few columns from the SNOW dump will be retained for future use. Please modify this list as needed.
    input_df = input_df[output_cols3]
    input_df.reset_index(inplace=True, drop=True)

    # Rename columns for ease of use
    col_dict = {'Task type': 'Task_type'}
    input_df.rename(columns=col_dict, inplace=True)

    # make a copy to proceed further
    pre_step1_done_f = True
    process_df = input_df

    return process_df, pre_step1_done_f


def get_options_for_dd(input_df) -> Tuple[list]:
    dd_option_list = input_df['Task_type'].unique()

    return dd_option_list


def pre_processing_step_two(input_df) -> Tuple[pd.DataFrame, bool]:
    process_df = input_df
    # Add additional fields
    # Add month and Year in which the ticket is raised
    process_df['month_year'] = process_df['Opened'].map(get_month)

    # Prepare data for extracting Resolution details from Work Notes
    # Remove new line characters from Work Notes
    process_df['Work notes'].replace(r'\s+|\\n', ' ', regex=True, inplace=True)

    # If work notes is empty, set it to 'not available'
    process_df['Work notes'].fillna("Not Available", inplace=True)

    # Get Resolution Details from work notes and create a separate column in the Dataset
    start = "Resolution details:"
    end = "Resolution code:"
    # process_df['res_detail'] = process_df['Work notes'].map(extract_string(start,end))
    process_df['res_detail'] = process_df.apply(extract_string, start=start, end=end, axis=1)

    # Get Resolution Code from work notes and create a separate column in the Dataset
    start = "Resolution code:"
    end = "Resolve time:"
    # process_df['res_code'] = process_df['Work notes'].map(extract_string(start,end))
    process_df['res_code'] = process_df.apply(extract_string, start=start, end=end, axis=1)

    # Get Resolution Time from work notes and create a separate column in the Dataset
    start = "Resolve time:"
    end = "; Resolved by:"
    # process_df['res_time'] = process_df['Work notes'].map(extract_string(start,end))
    process_df['res_time'] = process_df.apply(extract_string, start=start, end=end, axis=1)

    # Get Resolved by from work notes and create a separate column in the Dataset
    start = "Resolved by:"
    end = '; Resolved:'
    # process_df['res_by'] = process_df['Work notes'].map(extract_string(start,end))
    process_df['res_by'] = process_df.apply(extract_string, start=start, end=end, axis=1)

    # Add a column to indicate if it is a user or system generated
    process_df.loc[process_df['Opened by'] != 'Netcool Integration', 'dfident_source'] = 'user'
    process_df.loc[process_df['Opened by'] == 'Netcool Integration', 'dfident_source'] = 'System'

    process_done_f = True

    return process_df, process_done_f


def process_df_for_nlp(input_dataframe, weights_from_st, selected_option, eps, min_samples):
    # assign other processing variables passed from calling function
    output_cols2 = ['Task_type', 'Number', 'Group_B', 'month_year', 'Description', 'res_detail',
                    'res_code', 'res_time', 'res_by', 'Assignment group', 'State', 'Priority',
                    'Opened', 'Opened by', 'Assigned to', 'Closed', 'Application Criticality', 'Closed by',
                    'SLA Due Date']
    subtext = "Description"
    eps = eps
    min_samples = min_samples
    process_df = input_dataframe
    selected_option = selected_option
    # filter the df for further processing based on user selection
    filtered_df = process_df[process_df['Task_type'] == selected_option]
    filtered_df.reset_index(inplace=True, drop=True)
    filtered_df['rev_desc'] = ''
    for i in range(0, len(filtered_df) - 1):
        filtered_df['rev_desc'].iloc[i] = (find_substring(filtered_df['Description'].iloc[i], subtext))
    filtered_df['rev_desc'] = filtered_df['rev_desc'].replace(r'\n', ' ', regex=True)
    filtered_df['rev_desc'] = filtered_df['rev_desc'].astype(str)
    filtered_df['rev_desc'] = filtered_df['rev_desc'].str.replace('\W', ' ')
    filtered_df['rev_desc'] = filtered_df['rev_desc'].str.replace(' +', ' ')
    # Build encodings
    ticket_description = filtered_df['rev_desc'].to_list()
    ticket_encodings = get_encodings(weights_from_st, ticket_description)
    ticket_grouping = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(ticket_encodings)
    # Grouping of tickets
    # Get Group labels
    ticket_labels = ticket_grouping.labels_
    # Add Group as a new column
    filtered_df['Group_B'] = pd.DataFrame(ticket_labels)
    filtered_df = filtered_df[output_cols2]
    return filtered_df


def create_summary_for_df(input_df) -> Tuple[pd.DataFrame]:
    pivot_cols2 = ['Group_B', 'All']
    month_year_list = list(input_df.month_year.unique())
    pivot_cols2.extend(month_year_list)
    # create summary from the input data
    input_summary_df = pd.pivot_table(input_df, index=["Group_B"], columns=["month_year"], values=["Number"],
                                      aggfunc=[len], fill_value='0', margins=True)
    input_summary_df = pd.DataFrame(input_summary_df.to_records())
    input_summary_df.columns = [hdr.replace("'", "") for hdr in input_summary_df.columns]
    input_summary_df.columns = [hdr.replace(",", "") for hdr in input_summary_df.columns]
    input_summary_df.columns = [hdr.replace(")", "") for hdr in input_summary_df.columns]
    input_summary_df.columns = [hdr.replace("(", "") for hdr in input_summary_df.columns]
    input_summary_df.columns = [hdr.replace("len", "") for hdr in input_summary_df.columns]
    input_summary_df.columns = [hdr.replace("Number", "") for hdr in input_summary_df.columns]
    input_summary_df.columns = [hdr.strip() for hdr in input_summary_df.columns]
    #         vars()[temp2] = vars()[temp2].ix[:, pivot_cols2]
    input_summary_df = input_summary_df.loc[:, pivot_cols2]
    input_summary_df.rename(columns={'All': 'Total'}, inplace=True)
    input_summary_df = input_summary_df.sort_values(['Total'], ascending=False)
    input_summary_df = input_summary_df[input_summary_df['Group_B'] != 'All']
    input_summary_df.reset_index(inplace=True, drop=True)
    return input_summary_df


def main():
    # start of the App

    # Load the image and title
    load_header = load_page_head_with_image()
    if not load_header:
        st.error("Error loading the Header, Contact admin")

    # load the side bars
    load_side_bar, eps_inc, min_samples_inc, eps_task, min_samples_task = load_side_bar_with_options()
    if not load_side_bar:
        st.error("Error loading side bar & retrieving options, Contact admin")

    # request to upload the file
    upload_file, exception = upload_file_for_processing()
    if upload_file is None:
        st.error("error uploading the file, Upload the file")
    else:
        # read the file
        input_df, input_read_f = read_input_file(upload_file)
        if not input_read_f:
            st.error("Error in reading the file, check the input again")
        else:
            # Filter the data for tickets which were created on or after Jan 1, 2020
            input_df, date_filter_done_f = filter_based_on_date(input_df)
            if not date_filter_done_f:
                st.error("Error in filtering the data based on date, pls check")

            # This function will do step-1 preprocessing
            pre_process_one_df, pre_step1_done_f = pre_processing_step_one(input_df)
            if not pre_step1_done_f:
                st.error("Error in pre-processing step-1, pls check")

            # This function will do the step-2 preprocessing
            pre_process_two_df, pre_step2_done_f = pre_processing_step_two(pre_process_one_df)
            if not pre_step2_done_f:
                st.error("Error in pre-processing step-2, pls check")

            # get the options to be displayed in dropdown
            list_of_dd_options = get_options_for_dd(pre_process_two_df)
            # Load the model weights
            my_weights = get_weights()
            weights_from_st = copy.deepcopy(my_weights)

            # display the option to user to select from dropdown
            st.write('-' * 15)
            option_selected: object = st.selectbox("Select the category to proceed further", list_of_dd_options)
            st.write('-' * 15)

            # process based on user selection
            if option_selected != 'Incident':
                eps = eps_task
                min_samples = min_samples_task
                nlp_processed_df = process_df_for_nlp(pre_process_two_df, weights_from_st, option_selected, eps,
                                                      min_samples)
                st.write('Processed and Categorized data for selected Category')

                if nlp_processed_df.shape[0] > 0:
                    st.write(nlp_processed_df)
                    st.markdown(get_table_download_link(nlp_processed_df), unsafe_allow_html=True)
                    st.write('-' * 15)
                    nlp_processed_df = nlp_processed_df[nlp_processed_df['Group_B'] != -1]
                    if nlp_processed_df.shape[0] > 0:
                        nlp_data_summary_df = create_summary_for_df(nlp_processed_df)
                        if nlp_data_summary_df.shape[0] > 0:
                            st.write('Data Summary for selected Category')
                            st.write(nlp_data_summary_df)
                            st.markdown(get_table_download_link(nlp_data_summary_df), unsafe_allow_html=True)
                            st.write('-' * 15)
                        else:
                            st.warning("No Data available to Display")
                else:
                    st.write("No Data available to Display")
            elif option_selected == 'Incident':
                eps = eps_inc
                min_samples = min_samples_inc
                nlp_processed_df = process_df_for_nlp(pre_process_two_df, weights_from_st, option_selected, eps,
                                                      min_samples)
                st.write('Processed and Categorized data for selected Category')

                if nlp_processed_df.shape[0] > 0:
                    st.write(nlp_processed_df)
                    st.markdown(get_table_download_link(nlp_processed_df), unsafe_allow_html=True)
                    st.write('-' * 15)
                    nlp_processed_df = nlp_processed_df[nlp_processed_df['Group_B'] != -1]
                    if nlp_processed_df.shape[0] > 0:
                        nlp_data_summary_df = create_summary_for_df(nlp_processed_df)
                        if nlp_data_summary_df.shape[0] > 0:
                            st.write('Data Summary for selected Category')
                            st.write(nlp_data_summary_df)
                            st.markdown(get_table_download_link(nlp_data_summary_df), unsafe_allow_html=True)
                            st.write('-' * 15)
                        else:
                            st.warning("No Data available to Display")
                else:
                    st.write("No Data available to Display")
            else:
                st.error("Select the option")


if __name__ == '__main__':
    main()
