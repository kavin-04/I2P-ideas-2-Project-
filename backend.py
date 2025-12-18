from huggingface_hub import InferenceClient
import os

# ===============================
# 🔑 HUGGING FACE TOKEN
# ===============================
HF_TOKEN = os.environ.get("HF_TOKEN")

client = InferenceClient(token=HF_TOKEN)

# ===============================
# 🧠 MODEL CONFIG (BUSINESS OPTIMIZED)
# ===============================
MODEL_MAP = {
    # BUSINESS TASKS (Use Llama-3.1-8B - excellent for business/content)
    "business_strategy": "meta-llama/Llama-3.1-8B-Instruct",
    "pitch_deck": "meta-llama/Llama-3.1-8B-Instruct",
    "market_analysis": "meta-llama/Llama-3.1-8B-Instruct",
    "financial_model": "meta-llama/Llama-3.1-8B-Instruct",
    "business_plan": "meta-llama/Llama-3.1-8B-Instruct",
    "idea": "meta-llama/Llama-3.1-8B-Instruct",
    "guide": "meta-llama/Llama-3.1-8B-Instruct",
    "assistant": "meta-llama/Llama-3.1-8B-Instruct",
    "ppt": "meta-llama/Llama-3.1-8B-Instruct",
    "platform": "meta-llama/Llama-3.1-8B-Instruct",
    "flowchart": "meta-llama/Llama-3.1-8B-Instruct",
    
    # TECHNICAL TASKS (Use Qwen2.5-Coder-32B - best for code)
    "code": "Qwen/Qwen2.5-Coder-32B-Instruct",
}

TOKEN_MAP = {
    # Business tasks get more tokens for comprehensive responses
    "business_strategy": 1200,
    "pitch_deck": 1500,
    "market_analysis": 1200,
    "financial_model": 1000,
    "business_plan": 1800,
    "idea": 800,
    "guide": 1000,
    "ppt": 1200,
    "flowchart": 800,
    "platform": 600,
    "code": 1500,  # Code needs more tokens
    "assistant": 800,
}

# ===============================
# 🧠 MEMORY (SINGLE SESSION)
# ===============================
LAST_TASK = None
LAST_PROMPT = None
LAST_OUTPUT = ""

# ===============================
# 🔍 ENHANCED TASK DETECTION (Business Specific)
# ===============================
def detect_tasks(text: str):
    text = text.lower()
    tasks = []

    # BUSINESS-SPECIFIC DETECTION (Enhanced)
    if any(word in text for word in ["business", "startup", "company", "venture", "enterprise", "corporate"]):
        tasks.append("business_strategy")
    if any(word in text for word in ["pitch", "investor", "deck", "presentation", "vc", "funding", "raise"]):
        tasks.append("pitch_deck")
    if any(word in text for word in ["market", "competitor", "analysis", "research", "trend", "industry"]):
        tasks.append("market_analysis")
    if any(word in text for word in ["financial", "revenue", "cost", "profit", "roi", "forecast", "budget", "valuation"]):
        tasks.append("financial_model")
    if any(word in text for word in ["plan", "strategy", "roadmap", "timeline", "executive", "proposal"]):
        tasks.append("business_plan")
    
    # EXISTING DETECTION (Keep but enhance)
    if "idea" in text or "concept" in text or "innovate" in text:
        tasks.append("idea")
    if "guide" in text or "mentor" in text or "advice" in text or "help" in text:
        tasks.append("guide")
    if any(word in text for word in ["code", "program", "developer", "software", "algorithm", "function"]):
        tasks.append("code")
    if any(word in text for word in ["flow", "architecture", "system", "diagram", "process"]):
        tasks.append("flowchart")
    if any(word in text for word in ["platform", "deploy", "host", "server", "cloud", "infrastructure"]):
        tasks.append("platform")
    if any(word in text for word in ["ppt", "slide", "presentation", "powerpoint", "deck"]):
        tasks.append("ppt")

    if not tasks:
        tasks.append("assistant")

    # Prioritize: max 2 tasks for better quality
    priority_order = ["pitch_deck", "business_strategy", "financial_model", 
                     "code", "market_analysis", "business_plan", 
                     "idea", "guide", "ppt", "flowchart", "platform", "assistant"]
    
    # Sort by priority and take top 2
    sorted_tasks = sorted(tasks, key=lambda x: priority_order.index(x) if x in priority_order else 999)
    return sorted_tasks[:2]

# ===============================
# 🧩 ENHANCED PROMPT BUILDER FOR BUSINESS
# ===============================
def build_prompt(task, user_input):
    prompts = {
        # BUSINESS-SPECIFIC PROMPTS
        "business_strategy": f"""As a senior business consultant, analyze this business scenario:

{user_input}

Provide a comprehensive business strategy including:
1. SWOT Analysis (Strengths, Weaknesses, Opportunities, Threats)
2. Competitive Advantage (Unique value proposition)
3. Go-to-Market Strategy (How to reach customers)
4. Revenue Model (How to make money)
5. Risk Assessment(Key risks and mitigations)
6. 90-Day Action Plan (Immediate next steps)

Format with clear sections and actionable insights.""",
        
        "pitch_deck": f"""Create an investor-ready pitch deck for:

{user_input}

Structure it as a 10-slide pitch deck:
1. Title Slide - Company name & tagline
2. Problem - What pain point are you solving?
3. Solution - Your product/service
4. Market Size - TAM, SAM, SOM with data
5. Business Model - How you make money
6. Traction - Current progress & milestones
7. Competition - Competitive landscape
8. Team - Key team members & advisors
9. Financials - 3-year projections
10. The Ask - Funding needed & use of funds

Make it data-driven and compelling for investors.""",
        
        "market_analysis": f"""Conduct a professional market analysis for:

{user_input}

Include:
• Target Market Segmentation (B2B/B2C, demographics, psychographics)
• Competitor Analysis (Direct & indirect competitors with SWOT for each)
• Market Trends (Current & future trends affecting this space)
• Barriers to Entry (What makes this market difficult to enter)
• Growth Potential (Market growth rate & projections)
• Customer Acquisition Costs (Estimated CAC for this market)

Use real data references where possible.""",
        
        "financial_model": f"""Create financial projections for:

{user_input}

Provide:
1. Revenue Projections (3 years, monthly for year 1)
2. Cost Structure(Fixed vs variable costs)
3. Profit & Loss Statement (Projected P&L)
4. Cash Flow Forecast(Monthly cash flow)
5. Key Metrics (CAC, LTV, MRR/ARR if applicable)
6. Break-even Analysis (When the business becomes profitable)
7. Funding Requirements (How much capital needed)

Use realistic assumptions and explain each calculation.""",
        
        "business_plan": f"""Develop a comprehensive business plan for:

{user_input}

Structure:
Executive Summary (1-page overview)
Company Description (Mission, vision, legal structure)
Market Analysis (Industry, target market, competition)
Organization & Management (Team structure, roles)
Product/Service Line (What you're selling)
Marketing & Sales Strategy (How you'll acquire customers)
Funding Request (If seeking investment)
Financial Projections (3-5 year projections)
Appendix (Supporting documents)

Make it investor-ready and thorough.""",
        
        # ENHANCE EXISTING PROMPTS
        "idea": f"""Generate innovative, FUNDABLE business ideas for:

{user_input}

For each idea, provide:
• Problem Statement (What problem does it solve?)
• Solution Overview (How it works)
• Target Market (Who will pay for this?)
• Revenue Model (How to monetize)
• Competitive Advantage (Why this beats existing solutions)
• Initial MVP (Minimum viable product to test)
• Required Team (Key roles needed)
• Potential Challenges (Risks to consider)

Focus on ideas with real business potential.""",
        
        "guide": f"""Act as a Y Combinator-style mentor providing actionable guidance for:

{user_input}

Provide step-by-step mentorship:
1. Immediate Next Steps (What to do in next 7 days)
2. Key Milestones (Quarterly goals for next year)
3. Common Pitfalls (What to avoid in this space)
4. Resource Recommendations (Books, tools, networks)
5. Skill Development (What skills to build)
6. Networking Strategy (Who to connect with)
7. Success Metrics (How to measure progress)

Be specific and actionable, not generic advice.""",
        
        "code": (
            "You are a CTO/tech lead at a startup. Write production-ready, scalable code.\n\n"
            f"Business Requirement: {user_input}\n\n"
            "Provide:\n"
            "1. Complete Implementation (Clean, commented, tested code)\n"
            "2. Architecture Explanation (Why this approach?)\n"
            "3. Scalability Considerations (How it handles growth)\n"
            "4. Cost Optimization (Cloud costs, efficiency)\n"
            "5. Security Best Practices (Vulnerabilities to avoid)\n"
            "6. Deployment Instructions (How to deploy)\n"
            "7. Maintenance Plan (Ongoing maintenance needs)\n\n"
            "Write as if this code will be reviewed by senior engineers."
        ),
        
        # Keep existing but enhance
        "ppt": (
            "Create an INVESTOR-READY PowerPoint presentation with these strict guidelines:\n"
            "RULES:\n"
            "- Each slide MUST have a clear title\n"
            "- Use bullet points ONLY (no paragraphs)\n"
            "- Include data points & metrics where possible\n"
            "- Add speaker notes for each slide\n"
            "- Follow this exact structure:\n\n"
            "Slide 1: Title & Executive Summary\n"
            "Slide 2: The Problem (Quantify the pain)\n"
            "Slide 3: Our Solution (How we solve it)\n"
            "Slide 4: Market Opportunity (TAM/SAM/SOM)\n"
            "Slide 5: Business Model (Revenue streams)\n"
            "Slide 6: Traction & Milestones\n"
            "Slide 7: Competitive Landscape\n"
            "Slide 8: The Team (Why us?)\n"
            "Slide 9: Financial Projections\n"
            "Slide 10: The Ask & Use of Funds\n\n"
            f"Topic: {user_input}"
        ),
        
        "flowchart": (
            "Design a SYSTEM ARCHITECTURE diagram for a scalable business solution:\n\n"
            f"System Requirements: {user_input}\n\n"
            "Provide:\n"
            "1. High-Level Architecture (Components & interactions)\n"
            "2. Data Flow Diagram (How data moves through system)\n"
            "3. Technology Stack (Specific technologies recommended)\n"
            "4. Scalability Design (How it scales with users/traffic)\n"
            "5. Security Layer (Security measures at each level)\n"
            "6. Cost-Effective Design (Optimizing for business budgets)\n"
            "7. Deployment Architecture (Cloud/on-prem/hybrid)\n\n"
            "Use clear notation: [Component] → (Process) → {Database}"
        ),
        
        "platform": (
            "Recommend DEPLOYMENT PLATFORMS for a BUSINESS APPLICATION:\n\n"
            f"Application Details: {user_input}\n\n"
            "For EACH platform recommendation, provide:\n"
            "• Platform Name (AWS/GCP/Azure/Vercel/etc.)\n"
            "• Pricing Tier (Free/Startup/Enterprise costs)\n"
            "• Setup Complexity (Easy/Medium/Hard)\n"
            "• Best For (What use cases it's ideal for)\n"
            "• Scalability (How well it scales)\n"
            "• Limitations (What to watch out for)\n"
            "• Migration Path (How to move from free to paid)\n\n"
            "Prioritize platforms with good free tiers for startups."
        ),
        
        "assistant": (
            "You are I2P Business AI, an expert business consultant and technical advisor. "
            "Provide comprehensive, actionable business advice.\n\n"
            f"User Query: {user_input}\n\n"
            "Structure your response with:\n"
            "• Key Insights (Main takeaways)\n"
            "• Actionable Recommendations (Specific things to do)\n"
            "• Potential Risks (What to watch out for)\n"
            "• Next Steps (Immediate follow-up actions)\n"
            "• Resource Links (Tools, templates, further reading)"
        ),
    }
    return prompts.get(task, user_input)

# ===============================
# 🤖 MODEL CALL
# ===============================
def call_model(model_id, prompt, max_tokens):
    # Use text generation instead of chat completions
    response = client.text_generation(
        prompt=prompt,
        model=model_id,
        max_new_tokens=max_tokens,
        temperature=0.6,
        stream=False,
        details=False,
    )
    return response

# ===============================
# ✂️ TRUNCATION CHECK
# ===============================
def is_truncated(text: str):
    # Check if the text ends abruptly
    return not text.strip().endswith((".", "!", "?", "```", "}", "]"))

# ===============================
# 🧾 BUSINESS TEMPLATE FORMATTER
# ===============================
def format_business_output(task, raw_output):
    """Format AI output into business-ready templates"""
    
    templates = {
        "pitch_deck": f"""
🎯 INVESTOR PITCH DECK TEMPLATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{raw_output}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Pro Tip: Add actual metrics, customer testimonials, and team bios.
📊 Design: Use Canva/Pitch.com for professional design.
⏱️ Timing: Practice to deliver in 10 minutes.
""",
        
        "financial_model": f"""
💰 FINANCIAL MODEL SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{raw_output}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 Next Steps:
1. Input these numbers into an Excel/Google Sheets template
2. Create monthly cash flow projections
3. Calculate break-even point
4. Prepare for investor due diligence
""",
        
        "business_plan": f"""
📋 BUSINESS PLAN OUTLINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{raw_output}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Checklist:
□ Executive Summary (1 page)
□ Company Description (2 pages)
□ Market Analysis (3-5 pages)
□ Organization Structure (1-2 pages)
□ Product/Service Details (2-3 pages)
□ Marketing Strategy (2-3 pages)
□ Financial Projections (3-5 pages)
□ Appendix (as needed)
""",
        
        "business_strategy": f"""
🎯 BUSINESS STRATEGY DOCUMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{raw_output}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 Implementation Priority:
1. Execute 90-Day Action Plan immediately
2. Schedule weekly strategy review meetings
3. Track KPIs mentioned above
4. Revisit strategy quarterly
""",
        
        "market_analysis": f"""
📊 MARKET ANALYSIS REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{raw_output}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔍 Research Next Steps:
1. Validate assumptions with customer interviews
2. Monitor competitor pricing changes monthly
3. Subscribe to industry newsletters
4. Attend relevant conferences/meetups
"""
    }
    
    return templates.get(task, raw_output)

# ===============================
# 🚀 MAIN RESPONSE FUNCTION (BUSINESS ENHANCED)
# ===============================
def generate_response(user_input, continue_generation=False):
    global LAST_TASK, LAST_PROMPT, LAST_OUTPUT

    try:
        # --- CONTINUE MODE ---
        if continue_generation and LAST_TASK and LAST_PROMPT:
            continuation_prompt = (
                f"Original Business Task: {LAST_PROMPT}\n\n"
                f"Text generated so far: {LAST_OUTPUT}\n\n"
                "CRITICAL FOR BUSINESS: Continue EXACTLY from where this stopped. "
                "Do NOT repeat. Add more actionable details, data points, or implementation steps."
            )

            output = call_model(
                MODEL_MAP[LAST_TASK],
                continuation_prompt,
                1000  # More tokens for continuation
            )

            LAST_OUTPUT += "\n" + output

            return {
                "tasks": {
                    "continuation": {
                        "output": output,
                        "truncated": is_truncated(output)
                    }
                },
                "completed": not is_truncated(output),
                "business_task": LAST_TASK
            }

        # --- NORMAL MODE ---
        tasks = detect_tasks(user_input)
        results = {}
        all_complete = True
        combined_text_for_memory = ""

        for task in tasks:
            prompt = build_prompt(task, user_input)
            output = call_model(
                MODEL_MAP[task],
                prompt,
                TOKEN_MAP[task]
            )
            
            # Format for business use
            formatted_output = format_business_output(task, output)

            results[task] = {
                "output": formatted_output,
                "truncated": is_truncated(output),
                "task_type": task,
                "model_used": MODEL_MAP[task].split('/')[-1]  # Show which model
            }

            combined_text_for_memory += output + "\n\n"
            
            # Save the last task details for potential continuation
            LAST_TASK = task
            LAST_PROMPT = prompt

            if is_truncated(output):
                all_complete = False

        LAST_OUTPUT = combined_text_for_memory.strip()

        return {
            "tasks": results,
            "completed": all_complete,
            "business_focus": True,
            "total_tasks": len(tasks),
            "primary_task": tasks[0] if tasks else None
        }

    except Exception as e:
        print(f"Business AI Error: {e}")
        return {
            "tasks": {
                "error": {
                    "output": f"Business advisory system temporarily unavailable.\nTechnical details: {str(e)[:100]}",
                    "truncated": False,
                    "task_type": "error"
                }
            },
            "completed": True,
            "error": str(e),
            "business_focus": True

        }

