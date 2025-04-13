from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import datetime
import openai
from dotenv import load_dotenv
import os


import streamlit as st

# --- Simple Password Auth with Styling & Session ---

PASSWORD = "Anu@1504"  # Change to your actual password

def check_password():
    if "password_correct" not in st.session_state:
        # Centered styling
        st.markdown("""
            <style>
            .centered {
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                height: 75vh;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="centered">', unsafe_allow_html=True)
        st.image("https://i.imgur.com/XHJg7MW.png", width=80)  # Optional: Your logo
        st.title("🔐 Mind Log Login")
        st.text_input("Enter your password", type="password", key="password")
        if st.button("Login"):
            if st.session_state["password"] == PASSWORD:
                st.session_state["password_correct"] = True
                del st.session_state["password"]
            else:
                st.error("🚫 Incorrect password.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    elif not st.session_state["password_correct"]:
        st.error("🚫 Incorrect password.")
        st.stop()

check_password()



load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# File to store logs
log_file = Path("memory_log.csv")

# Initialize or load data
if log_file.exists():
    df = pd.read_csv(log_file)
else:
    df = pd.DataFrame(columns=[
        "date", "time_block", "activity", "energy", "distraction", 
        "cognitive_load", "thoughts_ideas", "thoughts_type"
    ])
    df.to_csv(log_file, index=False)

st.title("🧠 Mind Recall Log — Daily Companion & Dashboard")

# ------------------ SECTION 1: DAILY LOG ------------------
st.header("🗓️ Log Your Day (8 AM - 10 PM)")

with st.form("log_form"):
    date = st.date_input("Date", datetime.date.today())
    time_block = st.selectbox("Time Block", [
        "8–10 AM", "10–12 PM", "12–2 PM", "2–4 PM", "4–6 PM", "6–8 PM", "8–10 PM"
    ])
    activity = st.text_area("What did you do in this block?")
    energy = st.slider("Energy / Mood Level", 1, 10, 5)
    distraction = st.slider("Distraction Level", 1, 10, 5)
    cognitive_load = st.selectbox("Cognitive Load", ["Low", "Medium", "High"])
    thoughts_ideas = st.text_area("Thoughts / Ideas / Summary")
    thoughts_type = st.selectbox("Tag it as:", ["Summary", "Idea", "Goal"])
    submitted = st.form_submit_button("Log Entry")

    if submitted:
        new_entry = {
            "date": str(date), "time_block": time_block, "activity": activity,
            "energy": energy, "distraction": distraction, "cognitive_load": cognitive_load,
            "thoughts_ideas": thoughts_ideas, "thoughts_type": thoughts_type
        }
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        df.to_csv(log_file, index=False)
        st.success("✅ Entry saved!")

# ------------------ SECTION 2: NEXT-DAY MEMORY SCAN ------------------
st.header("🧠 Next-Day Memory Scan")

yesterday = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
df["date"] = pd.to_datetime(df["date"])
yesterday_logs = df[df["date"] == pd.to_datetime(yesterday)]

if not yesterday_logs.empty:
    st.subheader(f"Did you remember these from {yesterday}?")
    for i, row in yesterday_logs.iterrows():
        st.markdown(f"**{row['time_block']}** — {row['activity']}")
        recall_key = f"recall_{i}"
        trigger_key = f"trigger_{i}"

        recall = st.radio("Did you remember this later on your own?", ["Yes", "No"], key=recall_key)
        if recall == "Yes":
            st.selectbox("What triggered it?", ["Self", "Reminder", "Person", "Emotion", "Other"], key=trigger_key)
else:
    st.info("No logs found for yesterday to review.")

# ------------------ SECTION 3: WEEKLY INSIGHTS DASHBOARD ------------------
st.header("📊 Weekly Insights")

# Filter last 7 days
last_7_days = datetime.datetime.today() - datetime.timedelta(days=7)
df_week = df[df["date"] >= last_7_days]

if df_week.empty:
    st.warning("Not enough data from the past 7 days to generate insights.")
else:
    avg_energy = df_week["energy"].mean()
    avg_distraction = df_week["distraction"].mean()
    entry_count = df_week.shape[0]

    st.metric("Total Entries", entry_count)
    st.metric("Avg Energy Level", f"{avg_energy:.2f}")
    st.metric("Avg Distraction Level", f"{avg_distraction:.2f}")

    # Energy trend
    st.subheader("Energy Trend")
    fig1, ax1 = plt.subplots()
    df_week.groupby("date")["energy"].mean().plot(marker='o', ax=ax1)
    ax1.set_ylabel("Energy")
    ax1.set_xlabel("Date")
    st.pyplot(fig1)

    # Distraction trend
    st.subheader("Distraction Trend")
    fig2, ax2 = plt.subplots()
    df_week.groupby("date")["distraction"].mean().plot(marker='o', color='orange', ax=ax2)
    ax2.set_ylabel("Distraction")
    ax2.set_xlabel("Date")
    st.pyplot(fig2)

    # Thoughts type chart
    st.subheader("Reflections This Week")
    type_counts = df_week["thoughts_type"].value_counts()
    st.bar_chart(type_counts)

    # Ideas list
    st.subheader("🧠 Ideas Logged")
    ideas = df_week[df_week["thoughts_type"] == "Idea"]["thoughts_ideas"]
    for i, idea in enumerate(ideas, 1):
        st.markdown(f"**{i}.** {idea}")

    # AI-style Summary
    st.subheader("🧠 Weekly Summary Suggestions")
    summary = []
    if avg_energy < 5:
        summary.append("Your average energy was lower than optimal. Try scheduling breaks or energy resets.")
    if avg_distraction > 6:
        summary.append("High distraction levels were noted. Consider reducing digital interruptions.")
    if "Goal" in df_week["thoughts_type"].values:
        summary.append("You've logged goals. Consider reviewing progress weekly.")
    if "Idea" in df_week["thoughts_type"].values:
        summary.append("You've captured several ideas — revisit and prioritize them.")

    if not summary:
        summary.append("No significant trends detected. Keep logging consistently for better insights.")

    for point in summary:
        st.markdown(f"- {point}")

# Optional: Show all data
if st.checkbox("📂 Show All Log Data"):
    st.dataframe(df)
# ------------------ SECTION 4: AI MEMORY COACH BRIEFING ------------------
st.header("🧠 Your Weekly Cognitive Briefing by Memory Coach")

# AI Memory Coach Summary (Mocked Logic)
def generate_mock_memory_coach_summary(data):
    summary = []

    avg_energy = data["energy"].mean()
    avg_distraction = data["distraction"].mean()
    idea_count = data[data["thoughts_type"] == "Idea"].shape[0]
    goal_count = data[data["thoughts_type"] == "Goal"].shape[0]

    if avg_energy < 5:
        summary.append("You've had a low-energy week. Consider micro-breaks or adding something physical to your afternoon routine.")
    else:
        summary.append("Your energy was reasonably stable. Great job staying balanced.")

    if avg_distraction > 6:
        summary.append("Distractions seem high. You might benefit from silencing notifications or using a focus timer during key blocks.")
    else:
        summary.append("Focus levels were manageable this week.")

    if goal_count > 0:
        summary.append("You’ve set some goals — nice! Try reviewing them at the start and end of each day to anchor your intent.")

    if idea_count > 0:
        summary.append("Several new ideas popped up this week. Make time to revisit and act on the top 1–2 that energize you.")

    summary.append("Keep logging consistently. The more we map your mind, the better your memory muscle will perform!")

    return summary

# Only show if we have data
if df_week.empty:
    st.info("Not enough logs this week for your memory coach to comment yet.")
else:
    feedback = generate_mock_memory_coach_summary(df_week)
    for point in feedback:
        st.markdown(f"- {point}")
# ------------------ SECTION 5: GPT-4 Memory Coach Summary ------------------
import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.header("🧠 GPT-4 Memory Coach Summary")

# Define the GPT prompt function (no indent)
def build_gpt_prompt(df):
    csv_data = df.to_csv(index=False)
    prompt = f"""
Here is a week's worth of structured self-logging data from a user with aphantasia and SDAM.

They track:
- energy
- distraction
- cognitive load
- memory recall
- qualitative thoughts labeled as ideas, goals, and summaries

Here’s the data:
{csv_data}

Please analyze:
1. Trends in energy and distraction — were there spikes, dips, or steady zones?
2. Any overload or pacing issues with cognitive load?
3. Themes in the qualitative thoughts — any standout ideas or repeated patterns?
4. Possible links between energy, distraction, and memory clarity?
5. What seems to help (or hurt) their clarity and memory?
6. Suggest 2–3 small weekly experiments to try for better cognitive performance.

Return the analysis in 3 sections:
1. “Your Mental Map This Week”
2. “What Your Mind is Telling Us”
3. “Try This Next Week”

Use a friendly, coach-like voice — warm, observant, and practical.
"""
    return prompt

# Run the GPT summary if button is clicked
if st.button("🧠 Generate Weekly Summary"):
    with st.spinner("Thinking like your brain..."):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a cognitive performance coach helping the user reflect on mental clarity, memory, and ideas."},
                    {"role": "user", "content": build_gpt_prompt(df_week)}
                ],
                temperature=0.7
            )
            gpt_summary = response.choices[0].message["content"]
            st.markdown("### 📝 AI Memory Coach Summary")
            st.markdown(gpt_summary)
        except Exception as e:
            st.error(f"Something went wrong: {e}")

if not df_week.empty and "gpt_summary" in locals():
    pdf_file = generate_weekly_pdf(df_week, gpt_summary)
    st.download_button(
        label="📥 Download Weekly PDF Report",
        data=pdf_file,
        file_name="mind_recall_weekly_report.pdf",
        mime="application/pdf"
    )
# ------------------ SECTION 6: PDF Report Generation ------------------

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from io import BytesIO
import matplotlib.pyplot as plt

def generate_chart_image(dataframe, metric, title):
    fig, ax = plt.subplots()
    dataframe.groupby("date")[metric].mean().plot(marker='o', ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(metric.capitalize())
    buf = BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

def generate_reflection_chart(dataframe):
    counts = dataframe["thoughts_type"].value_counts()
    fig, ax = plt.subplots()
    counts.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title("Reflections by Type")
    ax.set_ylabel("Count")
    ax.set_xlabel("Type")
    buf = BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

def generate_weekly_pdf(dataframe, gpt_summary):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []

    styles = getSampleStyleSheet()
    heading = styles["Heading1"]
    body = styles["BodyText"]
    subtitle = ParagraphStyle(name="Subtitle", parent=styles["Heading2"], spaceAfter=10)

    # Cover Page
    elements.append(Paragraph("🧠 Mind Recall Weekly Report", heading))
    date_range = f"{dataframe['date'].min().date()} to {dataframe['date'].max().date()}"
    elements.append(Paragraph(f"Date Range: {date_range}", subtitle))
    elements.append(Paragraph("This report summarizes your cognitive activity, reflections, and memory trends over the past 7 days.", body))
    elements.append(PageBreak())

    # Charts Page
    energy_chart = generate_chart_image(dataframe, "energy", "Energy Over Time")
    distraction_chart = generate_chart_image(dataframe, "distraction", "Distraction Over Time")
    reflections_chart = generate_reflection_chart(dataframe)

    elements.append(Paragraph("📊 Visual Trends", heading))
    elements.append(Image(energy_chart, width=5.5*inch, height=3*inch))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Image(distraction_chart, width=5.5*inch, height=3*inch))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Image(reflections_chart, width=5.5*inch, height=3*inch))
    elements.append(PageBreak())

    # AI Summary Page
    elements.append(Paragraph("🧠 AI Memory Coach Summary", heading))
    for line in gpt_summary.split("\n"):
        elements.append(Paragraph(line.strip(), body))
    elements.append(PageBreak())

    # Thoughts & Ideas Page
    elements.append(Paragraph("💡 Thoughts, Ideas & Goals", heading))
    for t_type in ["Idea", "Goal", "Summary"]:
        entries = dataframe[dataframe["thoughts_type"] == t_type]["thoughts_ideas"]
        if not entries.empty:
            elements.append(Paragraph(f"{t_type}s", subtitle))
            for i, thought in enumerate(entries, 1):
                elements.append(Paragraph(f"{i}. {thought}", body))
            elements.append(Spacer(1, 0.2*inch))

    doc.build(elements)
    buffer.seek(0)
    return buffer

