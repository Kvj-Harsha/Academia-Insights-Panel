import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import utils

from rich.diagnose import report
from sklearn.linear_model import LinearRegression

# Function to calculate Grade based on marks
def calculate_grade(marks):
    if marks >= 90:
        return 'A+'
    elif marks >= 80:
        return 'A'
    elif marks >= 70:
        return 'B+'
    elif marks >= 60:
        return 'B'
    elif marks >= 50:
        return 'C+'
    elif marks >= 40:
        return 'C'
    elif marks >= 30:
        return 'D'
    else:
        return 'F'

# Function to convert grade to CGPA
def grade_to_cgpa(grade):
    grade_dict = {
        'A+': 10,
        'A': 9,
        'B+': 8,
        'B': 7,
        'C+': 6,
        'C': 5,
        'D': 4,
        'F': 0
    }
    return grade_dict.get(grade, 0)

# Load and process CSV file
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip().str.upper()  # Normalize column names
    return df

# Function to calculate statistics for each subject
def subject_analysis(df, subjects):
    result = {}
    for subject in subjects:
        scores = df[subject]
        grades = scores.apply(calculate_grade)
        cgpa = grades.apply(grade_to_cgpa)

        result[subject] = {
            'low': scores.min(),
            'mean': scores.mean(),
            'high': scores.max(),
            'top_scorers': df[df[subject] == scores.max()]['STUDENT NAME'].tolist(),
            'failed': df[df[subject] < 40]['STUDENT NAME'].tolist(),
            'pass_percentage': (len(df[df[subject] >= 40]) / len(df)) * 100,
            'mean_grade': calculate_grade(scores.mean())
        }
    return result

# Function to analyze individual student data
def student_analysis(df, student_info):
    student = df[df['STUDENT NAME'].str.contains(student_info, case=False) | df['ROLL NUMBER'].astype(str).str.contains(student_info)]
    if not student.empty:
        return student.iloc[0]
    else:
        return None

# Function to calculate CGPA for individual student
def calculate_student_cgpa(row, subjects):
    subject_grades = [grade_to_cgpa(calculate_grade(row[subject])) for subject in subjects]
    return np.mean(subject_grades)

# Export PDF function
def export_pdf(data, filename, image_path=r"C:\Users\harsh\OneDrive\rice cooker\Pictures\Screenshots\iiitr.png"):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Add page border
    border_margin = 20  # Set a margin for the border
    pdf.setLineWidth(1.5)
    pdf.rect(border_margin, border_margin, width - 2 * border_margin, height - 2 * border_margin)

    # Add image to top center if provided
    if image_path:
        try:
            img = utils.ImageReader(image_path)
            img_width, img_height = img.getSize()
            aspect_ratio = img_height / img_width

            # Set a maximum width for the image and adjust the height accordingly
            max_img_width = 200
            img_display_width = min(max_img_width, img_width)
            img_display_height = img_display_width * aspect_ratio

            img_x = (width - img_display_width) / 2
            img_y = height - 30 - img_display_height  # Keep space from the top for the title

            pdf.drawImage(image_path, img_x, img_y, width=img_display_width, height=img_display_height, preserveAspectRatio=True)
        except Exception as e:
            print(f"Error loading image: {e}")

    # Title
    title_y_position = height - 120  # Increased space between image and title
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(200, title_y_position, "Student Performance Report")

    # Set font for content
    pdf.setFont("Helvetica", 12)

    # Set starting position for content
    y_position = title_y_position - 40  # Increased space after title
    line_spacing = 18  # Increased line spacing for more gap between lines
    max_line_length = 100  # Maximum number of characters per line (for wrapping)

    for line in data:
        # Wrap text if it's too long
        while len(line) > max_line_length:
            wrapped_line = line[:max_line_length]
            pdf.drawString(30, y_position, wrapped_line)
            line = line[max_line_length:]
            y_position -= line_spacing

            # Start a new page if we're running out of space
            if y_position < border_margin + 50:
                pdf.showPage()
                pdf.setFont("Helvetica", 12)
                pdf.setLineWidth(1.5)
                pdf.rect(border_margin, border_margin, width - 2 * border_margin, height - 2 * border_margin)
                y_position = height - 50

        pdf.drawString(50, y_position, line)
        y_position -= line_spacing

        # Start a new page if we're running out of space
        if y_position < border_margin + 50:
            pdf.showPage()
            pdf.setFont("Helvetica", 12)
            pdf.setLineWidth(1.5)
            pdf.rect(border_margin, border_margin, width - 2 * border_margin, height - 2 * border_margin)
            y_position = height - 50

    pdf.save()

    buffer.seek(0)
    return buffer

def plot_attendance_impact(df, subjects):
    st.subheader("Impact of Attendance on Performance")

    # Dropdown to select subject
    selected_subject = st.selectbox("Select Subject", subjects)

    # Filter data for the selected subject
    attendance = df['ATTENDANCE (%)']
    scores = df[selected_subject]

    # Combined scatter plot with a linear trendline
    fig = px.scatter(df, x=attendance, y=scores, trendline="ols", title=f'Attendance vs {selected_subject} Scores')
    fig.update_layout(xaxis_title="Attendance (%)", yaxis_title=f'{selected_subject} Score')
    st.plotly_chart(fig)

    # Summary comments
    st.markdown("### Summary")

    # Calculate data-driven insights for summary
    num_students = len(df)

    # Low Attendance (<50%) vs Low Scores (<60%)
    low_attendance_students = df[df['ATTENDANCE (%)'] < 50]
    low_scores_students = df[df[selected_subject] < 60]
    low_attendance_low_scores = len(low_attendance_students[low_attendance_students[selected_subject] < 60])

    low_attendance_percentage = round((len(low_attendance_students) / num_students) * 100, 2) if len(low_attendance_students) > 0 else 0
    low_attendance_low_scores_percentage = round((low_attendance_low_scores / num_students) * 100, 2) if num_students > 0 else 0

    # Adding low attendance comment
    if low_attendance_percentage > 0:
        st.write(f"{low_attendance_percentage}% of students have attendance below 50%, "
                 f"and {low_attendance_low_scores_percentage}% of total students have both low attendance and low scores in {selected_subject}.")

    # Attendance Consistency (60-80%) and Average Scores
    mid_attendance_students = df[(df['ATTENDANCE (%)'] >= 60) & (df['ATTENDANCE (%)'] <= 80)]
    avg_score_mid_attendance = mid_attendance_students[selected_subject].mean() if len(mid_attendance_students) > 0 else 0

    mid_attendance_percentage = round((len(mid_attendance_students) / num_students) * 100, 2) if num_students > 0 else 0

    # Adding mid attendance comment
    if mid_attendance_percentage > 0:
        st.write(f"{mid_attendance_percentage}% of students have attendance between 60-80%, "
                 f"with an average score of {round(avg_score_mid_attendance, 2)} in {selected_subject}.")

    # Correlation Analysis
    correlation = df['ATTENDANCE (%)'].corr(df[selected_subject])
    if abs(correlation) < 0.3:  # Assuming low correlation as threshold < 0.3
        st.write(f"Attendance does not have a significant correlation with performance in {selected_subject}, with a correlation coefficient of {round(correlation, 2)}.")


# Streamlit application layout

st.title("Student Performance Analysis")
st.write("Welcome to the Student Performance Dashboard. Upload your student data in CSV format to begin analysis.")

# Upload CSV
csv_file = st.file_uploader("Upload CSV", type="csv")

if csv_file is not None:
    df = load_data(csv_file)

    st.write("Choose the type of analysis:")
    analysis_type = st.selectbox("Select Analysis Type", ["Overall Analysis", "Student-wise Analysis", "Attendance Impact Analysis", "Instructors"])

    subjects = ['CS103', 'CS121', 'CS211', 'EE121', 'ID151']

    if analysis_type == "Overall Analysis":
        st.subheader("Overall Class Analysis")

        # Calculate subject-wise analysis
        analysis_result = subject_analysis(df, subjects)

        # Attendance Gauge
        attendance_avg = df['ATTENDANCE (%)'].mean()
        st.write(f"### Average Attendance: {attendance_avg:.2f}%")

        # Plot Gauge for Attendance using Plotly
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=attendance_avg,
            title={'text': "Attendance Percentage"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#00cc96"},
                   'bgcolor': "lightgrey",
                   'steps': [
                       {'range': [0, 40], 'color': "red"},
                       {'range': [40, 70], 'color': "yellow"},
                       {'range': [70, 100], 'color': "green"}],
                   'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': attendance_avg}}))

        st.plotly_chart(fig)

        # Adding visual boxes around results
        for subject, result in analysis_result.items():
            cols = st.columns(2)  # 3 columns in one row

            st.markdown(
                """
                <style>
                .neon-card {
                    padding: 10px; 
                    margin: 5px; 
                    border-radius: 10px; 
                    background-color: #5C00A3; 
                    color: white; 
                    box-shadow: 0 0 5px white, 0 0 10px white, 0 0 20px #5C00A3, 0 0 30px #5C00A3;
                    transition: transform 0.3s, box-shadow 0.3s;
                }
                .neon-card:hover {
                    transform: scale(1.05);
                    box-shadow: 0 0 10px white, 0 0 20px white, 0 0 30px #5C00A3, 0 0 40px #5C00A3;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            with cols[0]:
                st.markdown(
                    f"<div class='neon-card'>"
                    f"<h4>{subject}</h4>"
                    f"<p><strong>Low Score:</strong> {result['low']}</p>"
                    f"<p><strong>Mean Score:</strong> {result['mean']} ({result['mean_grade']})</p>"
                    f"<p><strong>High Score:</strong> {result['high']}</p>"
                    f"<p><strong>Pass Percentage:</strong> {result['pass_percentage']:.2f}%</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            with cols[1]:
                st.markdown(
                    f"<div class='neon-card'>"
                    f"<p><strong>Top Scorers:</strong> {', '.join(result['top_scorers'])}</p>"
                    f"<p><strong>Failed Students:</strong> {', '.join(result['failed'])}</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            # Add a histogram for the subject scores
            fig_score_dist = go.Figure(
                data=[go.Histogram(x=df[subject], name='Score Distribution', marker_color='#007bff')])
            fig_score_dist.update_layout(
                title=f'{subject} Score Distribution',
                xaxis_title='Score',
                yaxis_title='Count',
                height=300,
                width=400,
                plot_bgcolor='rgba(0,0,0,0)',  # Transparent background for plot area
                paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for entire figure
                margin=dict(l=10, r=10, t=40, b=40),  # Margins around the plot
                font=dict(color='white')  # Set font color to white
            )

            # Use a container for the chart to apply the neon border
            st.markdown("<div class='neon-border'>", unsafe_allow_html=True)
            st.plotly_chart(fig_score_dist, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Display top CGPA
        df['CGPA'] = df[subjects].apply(
            lambda row: np.mean([grade_to_cgpa(calculate_grade(row[subject])) for subject in subjects]), axis=1)
        # Find the student(s) with the highest CGPA
        highest_cgpa_students = df[df['CGPA'] == df['CGPA'].max()][['STUDENT NAME', 'ROLL NUMBER']]

        # Extract student names and roll numbers
        student_names = highest_cgpa_students['STUDENT NAME'].tolist()
        roll_numbers = highest_cgpa_students['ROLL NUMBER'].tolist()

        # Combine student names with their roll numbers
        students_with_roll = [f"{name} (Roll No: {roll})" for name, roll in zip(student_names, roll_numbers)]

        with st.container():
            st.markdown(
                f"<div style='padding: 20px; border-radius: 10px; background-color: #5C00A3; color: white;'>"
                f"<h4>Highest CGPA: {df['CGPA'].max():.2f} by {', '.join(students_with_roll)}</h4>"
                f"</div>",
                unsafe_allow_html=True
            )

        st.title("")
        # Find the student(s) with the lowest CGPA
        lowest_cgpa_students = df[df['CGPA'] == df['CGPA'].min()][['STUDENT NAME', 'ROLL NUMBER']]

        # Extract student names and roll numbers
        student_names_lowest = lowest_cgpa_students['STUDENT NAME'].tolist()
        roll_numbers_lowest = lowest_cgpa_students['ROLL NUMBER'].tolist()

        # Combine student names with their roll numbers
        students_with_roll_lowest = [f"{name} (Roll No: {roll})" for name, roll in
                                     zip(student_names_lowest, roll_numbers_lowest)]

        with st.container():
            st.markdown(
                f"<div style='padding: 20px; border-radius: 10px; background-color: #5C00A3; color: white;'>"
                f"<h4>Lowest CGPA: {df['CGPA'].min():.2f} by {', '.join(students_with_roll_lowest)}</h4>"
                f"</div>",
                unsafe_allow_html=True
            )

        st.write("")

        # Calculate the High, Mean, and Low CGPAs
        highest_cgpa = df['CGPA'].max()
        mean_cgpa = df['CGPA'].mean()
        lowest_cgpa = df['CGPA'].min()

        # Create three columns for side-by-side alignment
        col1, col2, col3 = st.columns(3)

        with col1:
            # High CGPA Box
            st.markdown(
                f"<div style='padding: 20px; border-radius: 10px; background-color: #5C00A3; color: white;'>"
                f"<h4>High CGPA: {highest_cgpa:.2f}</h4>"
                f"</div>",
                unsafe_allow_html=True
            )

        with col2:
            # Mean CGPA Box
            st.markdown(
                f"<div style='padding: 20px; border-radius: 10px; background-color: #00A35C; color: white;'>"
                f"<h4>Mean CGPA: {mean_cgpa:.2f}</h4>"
                f"</div>",
                unsafe_allow_html=True
            )

        with col3:
            # Low CGPA Box
            st.markdown(
                f"<div style='padding: 20px; border-radius: 10px; background-color: #FF5733; color: white;'>"
                f"<h4>Low CGPA: {lowest_cgpa:.2f}</h4>"
                f"</div>",
                unsafe_allow_html=True
            )


    elif analysis_type == "Student-wise Analysis":
        st.subheader("Student-wise Analysis")

        student_info = st.text_input("Enter Student Name or Roll Number")
        if student_info:
            student_data = student_analysis(df, student_info)
            if student_data is not None:
                with st.container():
                    # Displaying student performance information with an image
                    # Displaying the student's image and performance details

                    st.markdown(
                        f"<div style='padding: 20px; border-radius: 10px; background-color: #5C00A3; color: white;'>"
                        f"<h4>Performance for {student_data['STUDENT NAME']}</h4>"
                        f"<p>Roll Number: {student_data['ROLL NUMBER']}</p>"
                        f"<p>Contact Number: {student_data['PHONE NO']}</p>"
                        f"<p>Email ID: {student_data['EMAIL ID']}</p>"
                        f"<p>Attendance: {student_data['ATTENDANCE (%)']}%</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                # Calculate and display subject scores and grades
                cols = st.columns(len(subjects))
                for i, subject in enumerate(subjects):
                    score = student_data[subject]
                    grade = calculate_grade(score)
                    with cols[i]:
                        st.markdown(
                            f"<div style='padding: 10px; margin: 5px; border-radius: 10px; background-color: #5C00A3; color: white;'>"
                            f"<h5>{subject}</h5>"
                            f"<p>Score: {score}</p>"
                            f"<p>Grade: {grade}</p>"
                            f"</div>", unsafe_allow_html=True)

                # Display overall CGPA
                cgpa = calculate_student_cgpa(student_data, subjects)
                st.write(f"**Overall CGPA**: {cgpa:.2f}")
            else:
                st.write("No student found with the provided information.")

    elif analysis_type == "Instructors":
        st.write("Check out all the Instructors!")
        last_four_columns = df.iloc[:5, -4:]
        st.dataframe(last_four_columns)

    elif analysis_type == "Attendance Impact Analysis":
        plot_attendance_impact(df, subjects)

    # PDF Export Section
    # PDF Export Section
    st.subheader("Export to PDF")
if st.button("Generate PDF Report"):
    report_lines = []

    if analysis_type == "Overall Analysis":
        report_lines.append("Overall Class Analysis Report")
        report_lines.append("")  # Double line break after title

        for subject, result in analysis_result.items():
            report_lines.append(f"{subject}:")
            report_lines.append(f"    Least Score: {result['low']}")
            report_lines.append(f"    Mean Score: {result['mean']} ({result['mean_grade']})")
            report_lines.append(f"    Highest Score: {result['high']}")
            report_lines.append(f"    Pass Percentage: {result['pass_percentage']:.2f}%")
            report_lines.append(f"    Top Scorers: {', '.join(result['top_scorers'])}")
            report_lines.append(f"    Failed Students: {', '.join(result['failed'])}")
            report_lines.append("")  # Double line break after each subject section

        report_lines.append(f"Average Attendance: {attendance_avg:.2f}%")
        report_lines.append(f"Mean CGPA: {mean_cgpa:.2f}")
        report_lines.append(f"Highest CGPA: {highest_cgpa:.2f}")
        report_lines.append("")  # Double line break at the end of the report

        pdf_file_name = "overall_analysis.pdf"

    elif analysis_type == "Student-wise Analysis" and student_data is not None:
        report_lines.append(f"Performance for {student_data['STUDENT NAME']}")
        report_lines.append(f"Roll Number: {student_data['ROLL NUMBER']}")
        report_lines.append(f"Contact number: {student_data['PHONE NO']}")
        report_lines.append(f"Email ID: {student_data['EMAIL ID']}")
        report_lines.append(f"Attendance: {student_data['ATTENDANCE (%)']}%")
        report_lines.append("")  # Double line break after student details

        for subject in subjects:
            score = student_data[subject]
            grade = calculate_grade(score)
            report_lines.append(f"{subject}: Score = {score}, Grade = {grade}")
            report_lines.append("")  # Double line break after each subject score

        report_lines.append(f"CGPA: {cgpa:.2f}")
        report_lines.append("")  # Double line break after CGPA
        pdf_file_name = f"{student_data['ROLL NUMBER']}.pdf"

    elif analysis_type == "Instructors":
        report_lines.append("Instructors List:")
        report_lines.append("")  # Double line break after title

        for index, row in last_four_columns.iterrows():
            report_lines.append(f"Instructor {index + 1}:")
            for col in last_four_columns.columns:
                report_lines.append(f"    {col}: {row[col]}")
            report_lines.append("")  # Double line break after each instructor

        pdf_file_name = "educators.pdf"

    elif analysis_type == "Attendance Impact Analysis":
        report_lines.append("Attendance Impact Analysis Report")
        report_lines.append(f"Subject: ")
        report_lines.append(f"Low Attendance: % of students")
        report_lines.append(f"Low Scores: % of students")
        report_lines.append(f"Correlation Coefficient: ")

        pdf_file_name = "attendence.pdf"

    # Generate and provide download link for the PDF
    pdf_buffer = export_pdf(report_lines, pdf_file_name,
                            image_path=r"C:\Users\harsh\OneDrive\rice cooker\Pictures\Screenshots\iiitr.png")
    st.download_button(label="Download PDF", data=pdf_buffer, file_name=pdf_file_name, mime="application/pdf")


