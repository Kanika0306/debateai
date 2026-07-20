"""
scripts/generate_eval_set.py — Generates the 30-claim evaluation dataset.
Writes to data/processed/eval_set.jsonl.
"""
import json
import os
from pathlib import Path

eval_set = [
    # ── True (8 claims) ──
    {
        "claim": "The number of new cases of shingles per year extends from 1.2 to 3.4 per 1,000.",
        "evidence": "Incidence rates of herpes zoster (shingles) range between 1.2 and 3.4 cases per 1,000 person-years.",
        "label": "True"
    },
    {
        "claim": "Gabrielle Union was in a movie.",
        "evidence": "Gabrielle Union is an American actress who starred in the films Bring It On and Bad Boys II.",
        "label": "True"
    },
    {
        "claim": "90 percent of Americans support universal background checks for gun purchases.",
        "evidence": "A national poll showed that 90% of American citizens support requiring background checks for all gun purchases.",
        "label": "True"
    },
    {
        "claim": "The World Bank is headquartered in Washington, D.C.",
        "evidence": "The World Bank Group is an international financial institution headquartered in Washington, D.C., USA.",
        "label": "True"
    },
    {
        "claim": "Global sea levels rose by about 8 inches in the last century.",
        "evidence": "NASA Vital Signs show that global sea levels rose about 8 inches (20 centimeters) in the last 100 years.",
        "label": "True"
    },
    {
        "claim": "NASA is an agency of the United States federal government.",
        "evidence": "The National Aeronautics and Space Administration (NASA) is an independent agency of the US federal government.",
        "label": "True"
    },
    {
        "claim": "The World Health Organization was founded in 1948.",
        "evidence": "The World Health Organization (WHO) is a specialized agency of the UN established on 7 April 1948.",
        "label": "True"
    },
    {
        "claim": "India's literacy rate was over 70 percent in the 2011 census.",
        "evidence": "According to the 2011 Census of India, the country's average literacy rate was recorded at 74.04 percent.",
        "label": "True"
    },

    # ── False (8 claims) ──
    {
        "claim": "Last year was one of the deadliest years ever for law enforcement officers.",
        "evidence": "According to official police statistics, last year actually saw a 10% decrease in line-of-duty officer fatalities.",
        "label": "False"
    },
    {
        "claim": "Schuyler VanValkenburg cosponsored a bill that would have allowed abortion until the moment of birth.",
        "evidence": "The bill in question only reduced the number of certifying doctors required for third-trimester abortions from three to one. It did not create a right to unconditional abortion up to birth.",
        "label": "False"
    },
    {
        "claim": "Nuclear power has caused millions of deaths worldwide.",
        "evidence": "Peer-reviewed studies indicate nuclear energy has one of the lowest mortality rates per unit of electricity generated, causing far fewer than 10,000 direct deaths in its entire history.",
        "label": "False"
    },
    {
        "claim": "The United States has never had a national debt.",
        "evidence": "The United States has carried a national debt since its founding in 1789, except for a brief period in 1835 under Andrew Jackson.",
        "label": "False"
    },
    {
        "claim": "COVID-19 is caused by a type of bacteria.",
        "evidence": "COVID-19 is an infectious disease caused by the SARS-CoV-2 virus, not bacteria.",
        "label": "False"
    },
    {
        "claim": "The Great Wall of China is the only man-made structure visible from the Moon with the naked eye.",
        "evidence": "Apollo astronauts confirmed that no human-made structures, including the Great Wall, are visible from the Moon without magnification.",
        "label": "False"
    },
    {
        "claim": "Average global temperatures have decreased over the last 50 years.",
        "evidence": "NASA climate data shows that the average global surface temperature has risen by approximately 1.0 degree Celsius (1.8 degrees Fahrenheit) since 1970.",
        "label": "False"
    },
    {
        "claim": "The World Bank only lends money to wealthy European countries.",
        "evidence": "The World Bank is a development bank that exclusively provides loans and grants to low- and middle-income countries.",
        "label": "False"
    },

    # ── Misleading (7 claims) ──
    {
        "claim": "Says Barack Obama robbed Medicare of $716 billion to pay for Obamacare.",
        "evidence": "Obamacare did cut $716 billion in future growth projections for Medicare spending over ten years. However, these cuts targeted provider payments and insurance subsidies to improve efficiency, not current beneficiary funds.",
        "label": "Misleading"
    },
    {
        "claim": "Under my administration, we have created more jobs than any president in history.",
        "evidence": "While job growth in absolute numbers was high due to post-pandemic recovery, the job growth rate as a percentage of the workforce was lower than several previous administrations.",
        "label": "Misleading"
    },
    {
        "claim": "The clean energy bill will raise your electric bill by 50 percent.",
        "evidence": "A worst-case scenario model predicted a temporary 10-15% increase, but the standard projection shows bills declining in the long term due to efficiency savings.",
        "label": "Misleading"
    },
    {
        "claim": "My opponent voted to cut the school district budget by $5 million.",
        "evidence": "The opponent voted against an expansion proposal of $5 million. The baseline school budget remained unchanged from the previous year.",
        "label": "Misleading"
    },
    {
        "claim": "Violent crime in the city doubled in the last year.",
        "evidence": "The number of reported violent crimes rose from 2 to 4 in a small precinct, which is a 100% increase but represents a statistically insignificant change overall.",
        "label": "Misleading"
    },
    {
        "claim": "The new vaccine has a 50 percent failure rate.",
        "evidence": "The vaccine has a 50% efficacy rate at preventing mild symptoms entirely, but it is 99% effective at preventing hospitalization and death.",
        "label": "Misleading"
    },
    {
        "claim": "We are spending $500 billion on foreign aid while our veterans go homeless.",
        "evidence": "The total U.S. foreign aid budget is approximately $50 billion, which represents less than 1% of the federal budget, not $500 billion.",
        "label": "Misleading"
    },

    # ── Unverified (7 claims) ──
    {
        "claim": "Eleveneleven was founded by a chef.",
        "evidence": "Eleveneleven is a fashion and lifestyle brand known for organic cotton apparel and hand-loomed fabrics. There is no public information regarding the founder's profession.",
        "label": "Unverified"
    },
    {
        "claim": "The governor secretly owns three offshore bank accounts in the Cayman Islands.",
        "evidence": "Public financial disclosure forms show standard domestic assets. There is no proof or official investigation confirming offshore accounts.",
        "label": "Unverified"
    },
    {
        "claim": "The new electric car model will be released exactly on December 1st.",
        "evidence": "The manufacturer stated they plan to launch the vehicle in 'late Q4', but has not announced a specific release date.",
        "label": "Unverified"
    },
    {
        "claim": "My opponent took bribes from representatives of the fossil fuel lobby.",
        "evidence": "Campaign finance records show public contributions from PACs, but there is no evidence or record of illegal bribery.",
        "label": "Unverified"
    },
    {
        "claim": "Local factories are dumping chemical waste into the river every midnight.",
        "evidence": "Environmental protection reports show standard emission levels during the day. No nighttime water quality testing has been conducted.",
        "label": "Unverified"
    },
    {
        "claim": "The mayor's chief of staff is planning to resign next month.",
        "evidence": "The chief of staff declined to comment on future employment plans, and no official announcement has been made.",
        "label": "Unverified"
    },
    {
        "claim": "A major earthquake will strike the city within the next forty-eight hours.",
        "evidence": "Seismologists state that it is currently impossible to predict the exact time and date of an earthquake with existing technology.",
        "label": "Unverified"
    }
]

def main():
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "eval_set.jsonl"
    
    with open(out_file, "w", encoding="utf-8") as f:
        for entry in eval_set:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Generated {len(eval_set)} eval examples at {out_file}")

if __name__ == "__main__":
    main()
