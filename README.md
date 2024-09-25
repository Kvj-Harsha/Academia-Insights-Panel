# 🎓 Student Performance Dashboard

A web-based application that allows users to analyze student performance data, visualize statistics, and generate personalized reports. This dashboard provides insights into student grades, CGPA, attendance impact on performance, and more, using interactive charts and visualizations.

## ✨ Features

- **📂 Upload CSV for Analysis**: Upload a CSV file containing student data (name, roll number, marks, attendance).
- **📊 Overall Class Analysis**: Get subject-wise analysis, including low, high, and average scores, pass percentages, and top scorers.
- **👤 Student-Wise Analysis**: Search for a specific student by name or roll number to view their detailed performance, including subject scores, grades, and CGPA.
- **📈 Attendance Impact Analysis**: Visualize the correlation between student attendance and performance using scatter plots with trendlines.
- **📝 PDF Report Export**: Generate and download a detailed PDF report based on the analysis.

## 🛠 Tech Stack

- **🐍 Python**: The main programming language used for logic and data processing.
- **🎨 Streamlit**: Framework used for building interactive web apps.
- **📊 Pandas**: For data manipulation and analysis.
- **📊 Plotly**: For interactive charts and visualizations.
- **📄 ReportLab**: For generating PDF reports.
- **📉 scikit-learn**: For linear regression in attendance impact analysis.

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/student-performance-dashboard.git
   cd student-performance-dashboard


1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/student-performance-dashboard.git
   cd student-performance-dashboard
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 💻 Usage

1. **Upload CSV File**: Upload a CSV file with the following structure:
   ```
   Roll Number, Student Name, CS103, CS121, CS211, EE121, ID151, Attendance (%)
   ```
2. **Select Analysis Type**: Choose between `Overall Analysis`, `Student-Wise Analysis`, or `Attendance Impact Analysis`.
3. **Generate Reports**: After analysis, you can generate a PDF report of the results.

### 📝 CSV Format Example

```csv
ROLL NUMBER, STUDENT NAME, CS103, CS121, CS211, EE121, ID151, ATTENDANCE (%)
101, John Doe, 85, 90, 78, 88, 92, 95
```
