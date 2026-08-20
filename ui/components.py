import html
import textwrap

import streamlit as st


# ============================================================
# GLOBAL THEME
# ============================================================

def apply_clinic_theme():
    """Apply the visual theme."""

    st.markdown(
        textwrap.dedent(
            """
            <style>

            .stApp {
                background-color: #f5f8fa;
            }

            .main .block-container {
                max-width: 900px;
                padding-top: 1.5rem;
                padding-bottom: 7rem;
            }

            #MainMenu {
                visibility: hidden;
            }

            footer {
                visibility: hidden;
            }

            header {
                background: transparent !important;
            }

            /* =========================
               CLINIC HEADER
               ========================= */

            .clinic-header {
                background: #ffffff;
                border: 1px solid #e3eaee;
                border-radius: 18px;
                padding: 20px 24px;
                margin-bottom: 28px;

                display: flex;
                align-items: center;
                justify-content: space-between;

                box-shadow: 0 3px 14px rgba(30, 70, 90, 0.06);
            }

            .clinic-brand {
                display: flex;
                align-items: center;
                gap: 14px;
            }

            .clinic-icon {
                width: 52px;
                height: 52px;
                border-radius: 14px;

                display: flex;
                align-items: center;
                justify-content: center;

                background: #e9f7f5;
                font-size: 27px;
            }

            .clinic-title {
                color: #183b4d;
                font-size: 21px;
                font-weight: 700;
                line-height: 1.2;
            }

            .clinic-subtitle {
                color: #718894;
                font-size: 13px;
                margin-top: 5px;
            }

            .online-status {
                display: flex;
                align-items: center;
                gap: 7px;

                color: #55717d;
                font-size: 13px;
            }

            .online-dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #20a875;
            }

            /* =========================
               BUTTONS
               ========================= */

            div.stButton > button {
                border: 1px solid #dce6ea;
                border-radius: 10px;
                background: #ffffff;
                color: #315765;
                font-size: 13px;
                min-height: 38px;
            }

            div.stButton > button:hover {
                border-color: #b8d4d5;
                color: #1d6d72;
            }

            /* =========================
               WELCOME
               ========================= */

            .welcome-card {
                background: #ffffff;
                border: 1px solid #e3eaee;
                border-radius: 18px;

                padding: 28px 24px;
                margin-bottom: 26px;

                box-shadow: 0 3px 14px rgba(30, 70, 90, 0.05);
            }

            .welcome-title {
                color: #183b4d;
                font-size: 22px;
                font-weight: 700;
                margin-bottom: 10px;
            }

            .welcome-text {
                color: #607985;
                font-size: 14px;
                line-height: 1.7;
                margin-bottom: 20px;
            }

            .service-list {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
            }

            .service-pill {
                display: inline-block;
                background: #eef8f7;
                color: #35666b;

                border-radius: 20px;
                padding: 8px 12px;

                font-size: 12px;
                font-weight: 500;
            }

            /* =========================
               USER MESSAGE
               ========================= */

            .user-message-row {
                display: flex;
                justify-content: flex-end;
                margin: 14px 0;
            }

            .user-message {
                max-width: 75%;

                background: #e4f3f2;
                color: #244f55;

                border-radius: 16px 16px 4px 16px;

                padding: 12px 16px;

                font-size: 14px;
                line-height: 1.6;
            }

            /* =========================
               ASSISTANT MESSAGE
               ========================= */

            .assistant-message-row {
                display: flex;
                justify-content: flex-start;
                margin: 14px 0;
            }

            .assistant-message {
                max-width: 82%;

                background: #ffffff;
                color: #455f69;

                border: 1px solid #e2eaed;

                border-radius: 16px 16px 16px 4px;

                padding: 14px 17px;

                font-size: 14px;
                line-height: 1.7;

                box-shadow: 0 2px 8px rgba(30, 70, 90, 0.04);
            }

            /* =========================
               DOCTOR CARD
               ========================= */

            .doctor-card {
                background: #ffffff;

                border: 1px solid #dce9eb;
                border-radius: 16px;

                padding: 18px;
                margin: 12px 0 18px 0;

                box-shadow: 0 3px 12px rgba(30, 70, 90, 0.05);
            }

            .doctor-name {
                color: #183b4d;
                font-size: 17px;
                font-weight: 700;
                margin-bottom: 5px;
            }

            .doctor-specialization {
                color: #34777a;
                font-size: 13px;
                font-weight: 600;
                margin-bottom: 13px;
            }

            .doctor-info {
                color: #617983;
                font-size: 13px;
                line-height: 1.7;
            }

            .doctor-label {
                color: #385965;
                font-weight: 600;
            }

            /* =========================
               FOOTER
               ========================= */

            .clinic-footer {
                text-align: center;

                color: #8ba0a9;

                font-size: 11px;
                line-height: 1.7;

                padding: 24px 10px 100px;
            }

            </style>
            """
        ),
        unsafe_allow_html=True,
    )


# ============================================================
# HEADER
# ============================================================

def render_clinic_header():
    """Render generic multi-clinic header."""

    st.markdown(
        textwrap.dedent(
            """
            <div class="clinic-header">

                <div class="clinic-brand">

                    <div class="clinic-icon">
                        🏥
                    </div>

                    <div>

                        <div class="clinic-title">
                            ClinicCare Assistant
                        </div>

                        <div class="clinic-subtitle">
                            Multi-clinic information assistant
                        </div>

                    </div>

                </div>

                <div class="online-status">
                    <span class="online-dot"></span>
                    Online
                </div>

            </div>
            """
        ),
        unsafe_allow_html=True,
    )


# ============================================================
# WELCOME
# ============================================================

def render_welcome():
    """Render welcome section."""

    st.markdown(
        textwrap.dedent(
            """
            <div class="welcome-card">

                <div class="welcome-title">
                    How can we help you today?
                </div>

                <div class="welcome-text">
                    I'm your AI clinic receptionist. I can help you
                    find doctors, check availability, explore clinic
                    services, and find clinic locations.
                </div>

                <div class="service-list">

                    <span class="service-pill">
                        🧑‍⚕️ Doctors
                    </span>

                    <span class="service-pill">
                        📅 Availability
                    </span>

                    <span class="service-pill">
                        🏥 Clinic services
                    </span>

                    <span class="service-pill">
                        📍 Locations
                    </span>

                    <span class="service-pill">
                        ☎️ Contact
                    </span>

                </div>

            </div>
            """
        ),
        unsafe_allow_html=True,
    )


# ============================================================
# USER MESSAGE
# ============================================================

def render_user_message(content):
    """Render user message."""

    safe_content = html.escape(str(content))

    st.markdown(
        textwrap.dedent(
            f"""
            <div class="user-message-row">

                <div class="user-message">
                    {safe_content}
                </div>

            </div>
            """
        ),
        unsafe_allow_html=True,
    )


# ============================================================
# ASSISTANT MESSAGE
# ============================================================

def render_assistant_message(content):
    """Render assistant message."""

    safe_content = html.escape(str(content))
    safe_content = safe_content.replace("\n", "<br>")

    st.markdown(
        textwrap.dedent(
            f"""
            <div class="assistant-message-row">

                <div class="assistant-message">
                    {safe_content}
                </div>

            </div>
            """
        ),
        unsafe_allow_html=True,
    )


# ============================================================
# DOCTOR CARD
# ============================================================

def render_doctor_card(
    doctor=None,
    specialization=None,
    clinic=None,
    availability=None,
):
    """Render doctor information."""

    if not doctor:
        return

    safe_doctor = html.escape(str(doctor))

    safe_specialization = html.escape(
        str(specialization or "Not specified")
    )

    safe_clinic = html.escape(
        str(clinic or "Not specified")
    )

    safe_availability = html.escape(
        str(availability or "Not specified")
    )

    st.markdown(
        textwrap.dedent(
            f"""
            <div class="doctor-card">

                <div class="doctor-name">
                    🧑‍⚕️ {safe_doctor}
                </div>

                <div class="doctor-specialization">
                    {safe_specialization}
                </div>

                <div class="doctor-info">

                    <div>
                        <span class="doctor-label">
                            🏥 Clinic:
                        </span>
                        {safe_clinic}
                    </div>

                    <div>
                        <span class="doctor-label">
                            📅 Availability:
                        </span>
                        {safe_availability}
                    </div>

                </div>

            </div>
            """
        ),
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

def render_footer():
    """Render footer."""

    st.markdown(
        textwrap.dedent(
            """
            <div class="clinic-footer">

                ClinicCare Assistant · Multi-clinic information assistant

                <br>

                For medical advice, please consult a qualified
                healthcare professional.

            </div>
            """
        ),
        unsafe_allow_html=True,
    )
