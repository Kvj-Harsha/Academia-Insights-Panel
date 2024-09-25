import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
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
def export_pdf(data, filename):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Title
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(200, height - 50, "Student Performance Report")

    pdf.setFont("Helvetica", 12)

    y_position = height - 100
    for line in data:
        pdf.drawString(50, y_position, line)
        y_position -= 20

    pdf.save()

    buffer.seek(0)
    return buffer

# Function to plot attendance impact
def plot_attendance_impact(df, subjects):
    st.subheader("Impact of Attendance on Performance")

    for subject in subjects:
        # Filter data for the subject
        attendance = df['ATTENDANCE (%)']
        scores = df[subject]

        # Scatter plot with a linear trendline
        fig = px.scatter(df, x=attendance, y=scores, trendline="ols", title=f'Attendance vs {subject} Scores')
        fig.update_layout(xaxis_title="Attendance (%)", yaxis_title=f'{subject} Score')
        st.plotly_chart(fig)

# Streamlit application layout
st.title("Student Performance Analysis")
st.write("Welcome to the Student Performance Dashboard. Upload your student data in CSV format to begin analysis.")

# Upload CSV
csv_file = st.file_uploader("Upload CSV", type="csv")

if csv_file is not None:
    df = load_data(csv_file)

    st.write("Choose the type of analysis:")
    analysis_type = st.selectbox("Select Analysis Type", ["Overall Analysis", "Student-wise Analysis", "Attendance Impact Analysis"])

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
            cols = st.columns(3)  # 3 columns in one row

            with cols[0]:
                st.markdown(
                    f"<div style='padding: 10px; margin: 5px; border-radius: 10px; background-color: #5C00A3; color: white;'>"
                    f"<h4>{subject}</h4>"
                    f"<p>Low Score: {result['low']}</p>"
                    f"<p>Mean Score: {result['mean']} ({result['mean_grade']})</p>"
                    f"</div>", unsafe_allow_html=True)

            with cols[1]:
                st.markdown(
                    f"<div style='padding: 10px; margin: 5px; border-radius: 10px; background-color: #5C00A3; color: white;'>"
                    f"<p>High Score: {result['high']}</p>"
                    f"<p>Pass Percentage: {result['pass_percentage']:.2f}%</p>"
                    f"</div>", unsafe_allow_html=True)

            with cols[2]:
                st.markdown(
                    f"<div style='padding: 10px; margin: 5px; border-radius: 10px; background-color: #5C00A3; color: white;'>"
                    f"<p>Top Scorers: {', '.join(result['top_scorers'])}</p>"
                    f"<p>Failed Students: {', '.join(result['failed'])}</p>"
                    f"</div>", unsafe_allow_html=True)

            # Add a histogram for the subject scores
            fig_score_dist = go.Figure(data=[go.Histogram(x=df[subject], name='Score Distribution', marker_color='#007bff')])
            fig_score_dist.update_layout(title=f'{subject} Score Distribution', xaxis_title='Score', yaxis_title='Count', height=300, width=400)
            st.plotly_chart(fig_score_dist, use_container_width=True)

        # Display top CGPA
        df['CGPA'] = df[subjects].apply(
            lambda row: np.mean([grade_to_cgpa(calculate_grade(row[subject])) for subject in subjects]), axis=1)

        highest_cgpa_student = df[df['CGPA'] == df['CGPA'].max()]['STUDENT NAME'].tolist()

        with st.container():
            st.markdown(
                f"<div style='padding: 20px; border-radius: 10px; background-color: #5C00A3; color: white;'>"
                f"<h4>Highest CGPA: {df['CGPA'].max():.2f} by {', '.join(highest_cgpa_student)}</h4>"
                f"<p>Instructor: {df['INSTRUCTER'].iloc[0]}</p>"
                f"</div>", unsafe_allow_html=True)

    elif analysis_type == "Student-wise Analysis":
        st.subheader("Student-wise Analysis")

        student_info = st.text_input("Enter Student Name or Roll Number")
        if student_info:
            student_data = student_analysis(df, student_info)
            if student_data is not None:
                with st.container():
                    st.markdown(
                        f"<div style='padding: 20px; border-radius: 10px; background-color: #5C00A3; color: white;'>"
                        f"<h4>Performance for {student_data['STUDENT NAME']}</h4>"
                        f"<p>Roll Number: {student_data['ROLL NUMBER']}</p>"
                        f"<p>Instructor: {student_data['INSTRUCTER']}</p>"
                        f"<p>Attendance: {student_data['ATTENDANCE (%)']}%</p>"
                        f"</div>", unsafe_allow_html=True)

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

    elif analysis_type == "Attendance Impact Analysis":
        plot_attendance_impact(df, subjects)

    # PDF Export Section
    st.subheader("Export to PDF")
    if st.button("Generate PDF Report"):
        report_lines = []
        if analysis_type == "Overall Analysis":
            report_lines.append(f"Overall Class Analysis Report")
            for subject, result in analysis_result.items():
                report_lines.append(f"{subject}:")
                report_lines.append(f"    Low Score: {result['low']}")
                report_lines.append(f"    Mean Score: {result['mean']} ({result['mean_grade']})")
                report_lines.append(f"    High Score: {result['high']}")
                report_lines.append(f"    Pass Percentage: {result['pass_percentage']:.2f}%")
                report_lines.append(f"    Top Scorers: {', '.join(result['top_scorers'])}")
                report_lines.append(f"    Failed Students: {', '.join(result['failed'])}")

            report_lines.append(f"Average Attendance: {attendance_avg:.2f}%")

        elif analysis_type == "Student-wise Analysis" and student_data is not None:
            report_lines.append(f"Performance for {student_data['STUDENT NAME']}")
            report_lines.append(f"Roll Number: {student_data['ROLL NUMBER']}")
            report_lines.append(f"Instructor: {student_data['INSTRUCTER']}")
            report_lines.append(f"Attendance: {student_data['ATTENDANCE (%)']}%")

            for subject in subjects:
                score = student_data[subject]
                grade = calculate_grade(score)
                report_lines.append(f"{subject}: Score = {score}, Grade = {grade}")

            report_lines.append(f"Overall CGPA: {cgpa:.2f}")

        # Generate and provide download link for the PDF
        pdf_buffer = export_pdf(report_lines, "student_report.pdf")
        st.download_button(label="Download PDF", data=pdf_buffer, file_name="student_report.pdf", mime="application/pdf")
