import streamlit as st


# ============================================================
# PAGE / GLOBAL STYLING
# ============================================================

def apply_clinic_theme():
    """
    Apply the visual theme for the clinic receptionist.
    """

    st.markdown(
        """
        <style>

        /* --------------------------------------------------
           GLOBAL
        -------------------------------------------------- */

        .stApp {
            background: #f5f8fa;
        }

        .main .block-container {
            max-width: 1000px;
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        /* Hide Streamlit default UI elements */
        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }

        header {
            visibility: hidden;
        }


        /* --------------------------------------------------
           CLINIC HEADER
        -------------------------------------------------- */

        .clinic-header {
            background: white;
            border: 1px solid #e5eaee;
            border-radius: 18px;
            padding: 18px 22px;
            margin-bottom: 18px;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);

            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .clinic-brand {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .clinic-logo {
            width: 48px;
            height: 48px;
            border-radius: 14px;
            background: #e8f5f3;

            display: flex;
            align-items: center;
            justify-content: center;

            font-size: 25px;
        }

        .clinic-name {
            font-size: 20px;
            font-weight: 700;
            color: #173b4d;
            margin: 0;
        }

        .clinic-subtitle {
            font-size: 13px;
            color: #71808a;
            margin-top: 3px;
        }

        .online-status {
            display: flex;
            align-items: center;
            gap: 7px;

            font-size: 13px;
            color: #587078;
        }

        .online-dot {
            width: 9px;
            height: 9px;
            border-radius: 50%;
            background: #28a879;
        }


        /* --------------------------------------------------
           WELCOME CARD
           -------------------------------------------------- */

        .welcome-card {
            background: white;
            border: 1px solid #e5eaee;
            border-radius: 18px;

            padding: 24px;
            margin-bottom: 20px;

            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.035);
        }

        .welcome-title {
            color: #173b4d;
            font-size: 21px;
            font-weight: 700;
            margin-bottom: 7px;
        }

        .welcome-text {
            color: #667780;
            font-size: 14px;
            line-height: 1.6;
            margin-bottom: 18px;
        }

        .service-list {
            display: flex;
            flex-wrap: wrap;
            gap: 9px;
        }

        .service-pill {
            background: #f0f7f6;
            color: #356b6a;

            border-radius: 999px;
            padding: 8px 13px;

            font-size: 13px;
            font-weight: 500;
        }


        /* --------------------------------------------------
           CHAT AREA
           -------------------------------------------------- */

        .chat-container {
            margin-top: 10px;
        }

        .user-message {
            display: flex;
            justify-content: flex-end;

            margin: 14px 0;
        }

        .user-bubble {
            max-width: 72%;

            background: #2d7f7a;
            color: white;

            padding: 12px 16px;
            border-radius: 17px 17px 4px 17px;

            font-size: 14px;
            line-height: 1.5;

            box-shadow: 0 2px 7px rgba(45, 127, 122, 0.14);
        }

        .assistant-message {
            display: flex;
            align-items: flex-start;
            gap: 10px;

            margin: 16px 0;
        }

        .assistant-avatar {
            width: 34px;
            height: 34px;

            flex-shrink: 0;

            border-radius: 11px;
            background: #e5f3f1;

            display: flex;
            align-items: center;
            justify-content: center;

            font-size: 17px;
        }

        .assistant-bubble {
            max-width: 78%;

            background: white;
            color: #344952;

            border: 1px solid #e3eaed;

            padding: 13px 16px;

            border-radius: 4px 17px 17px 17px;

            font-size: 14px;
            line-height: 1.6;

            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.025);
        }


        /* --------------------------------------------------
           DOCTOR INFORMATION CARD
           -------------------------------------------------- */

        .doctor-card {
            background: #ffffff;

            border: 1px solid #dfe9e9;
            border-radius: 15px;

            padding: 16px;
            margin-top: 10px;

            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.025);
        }

        .doctor-name {
            color: #173b4d;
            font-size: 16px;
            font-weight: 700;
        }

        .doctor-specialization {
            color: #71808a;
            font-size: 13px;
            margin-top: 3px;
        }

        .doctor-details {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;

            margin-top: 14px;
        }

        .doctor-detail {
            background: #f6f9fa;

            border-radius: 10px;

            padding: 9px 11px;

            font-size: 12px;
            color: #52656d;
        }


        /* --------------------------------------------------
           QUICK QUESTIONS
           -------------------------------------------------- */

        .quick-title {
            color: #65757d;
            font-size: 12px;
            font-weight: 600;

            margin-top: 20px;
            margin-bottom: 8px;
        }


        /* --------------------------------------------------
           FOOTER
           -------------------------------------------------- */

        .clinic-footer {
            text-align: center;

            color: #8a999f;
            font-size: 11px;

            margin-top: 22px;
            padding-top: 10px;
        }


        /* --------------------------------------------------
           STREAMLIT INPUT
           -------------------------------------------------- */

        div[data-testid="stChatInput"] {
            padding-bottom: 10px;
        }

        div[data-testid="stChatInput"] textarea {
            border-radius: 16px !important;
            border: 1px solid #dbe4e7 !important;

            background: white !important;

            padding: 12px 15px !important;

            font-size: 14px !important;
        }

        div[data-testid="stChatInput"] textarea:focus {
            border-color: #2d7f7a !important;
            box-shadow: 0 0 0 1px #2d7f7a !important;
        }


        /* --------------------------------------------------
           BUTTONS
           -------------------------------------------------- */

        .stButton > button {
            border-radius: 10px;
            border: 1px solid #dce5e7;

            background: white;
            color: #42616b;

            font-size: 13px;
        }

        .stButton > button:hover {
            border-color: #2d7f7a;
            color: #2d7f7a;
        }


        /* --------------------------------------------------
           MOBILE
           -------------------------------------------------- */

        @media (max-width: 700px) {

            .main .block-container {
                padding-left: 12px;
                padding-right: 12px;
            }

            .clinic-header {
                padding: 15px;
            }

            .clinic-name {
                font-size: 17px;
            }

            .online-status {
                display: none;
            }

            .welcome-card {
                padding: 18px;
            }

            .user-bubble,
            .assistant-bubble {
                max-width: 88%;
            }

        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# CLINIC HEADER
# ============================================================

def render_clinic_header(
    clinic_name: str = "Sunrise Medical Center",
):
    """
    Render the main clinic header.
    """

    st.markdown(
        f"""
        <div class="clinic-header">

            <div class="clinic-brand">

                <div class="clinic-logo">
                    🏥
                </div>

                <div>
                    <div class="clinic-name">
                        {clinic_name}
                    </div>

                    <div class="clinic-subtitle">
                        AI Clinic Receptionist
                    </div>
                </div>

            </div>

            <div class="online-status">
                <span class="online-dot"></span>
                Online
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# WELCOME CARD
# ============================================================

def render_welcome():
    """
    Render the initial receptionist welcome card.
    """

    st.markdown(
        """
        <div class="welcome-card">

            <div class="welcome-title">
                How can we help you today?
            </div>

            <div class="welcome-text">
                I'm your AI clinic receptionist. I can help
                you find doctors, check availability, and
                provide general clinic information.
            </div>

            <div class="service-list">

                <span class="service-pill">
                    👨‍⚕️ Doctors
                </span>

                <span class="service-pill">
                    📅 Availability
                </span>

                <span class="service-pill">
                    🏥 Clinic services
                </span>

                <span class="service-pill">
                    📍 Location
                </span>

                <span class="service-pill">
                    ☎️ Contact
                </span>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# USER MESSAGE
# ============================================================

def render_user_message(message: str):
    """
    Render a user chat bubble.
    """

    st.markdown(
        f"""
        <div class="user-message">
            <div class="user-bubble">
                {message}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ASSISTANT MESSAGE
# ============================================================

def render_assistant_message(message: str):
    """
    Render an AI receptionist chat bubble.
    """

    st.markdown(
        f"""
        <div class="assistant-message">

            <div class="assistant-avatar">
                🏥
            </div>

            <div class="assistant-bubble">
                {message}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DOCTOR CARD
# ============================================================

def render_doctor_card(
    doctor: str,
    specialization: str | None = None,
    clinic: str | None = None,
    availability: str | None = None,
):
    """
    Render structured doctor information.
    """

    specialization_html = ""

    if specialization:
        specialization_html = (
            f'<div class="doctor-specialization">'
            f'{specialization}'
            f'</div>'
        )

    details = []

    if clinic:
        details.append(f"🏥 {clinic}")

    if availability:
        details.append(f"🕐 {availability}")

    details_html = "".join(
        f'<div class="doctor-detail">{item}</div>'
        for item in details
    )

    st.markdown(
        f"""
        <div class="doctor-card">

            <div class="doctor-name">
                👨‍⚕️ {doctor}
            </div>

            {specialization_html}

            <div class="doctor-details">
                {details_html}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# QUICK QUESTIONS
# ============================================================

def render_quick_questions():
    """
    Render common questions above the chat input.
    """

    st.markdown(
        """
        <div class="quick-title">
            Popular questions
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

def render_footer():
    """
    Render a small professional footer.
    """

    st.markdown(
        """
        <div class="clinic-footer">
            AI Clinic Receptionist · Information assistant
            <br>
            For medical advice, please consult a qualified
            healthcare professional.
        </div>
        """,
        unsafe_allow_html=True,
    )