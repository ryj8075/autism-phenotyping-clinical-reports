#!/usr/bin/env Rscript
# Table 3 — 19-domain phenotype ontology.
rm(list = ls())
source("_common.R")

df <- data.frame(
  Code = c(
    "CO1", "CO2", "CO3", "CO4", "CO5", "CO6", "CO7",
    "AS1", "AS2", "AS3", "AS4", "AS5", "AS6", "AS7", "AS8", "AS9",
    "RE1", "RE2", "RE3"
  ),
  Group = c(
    rep("Core ASD domains", 7),
    rep("Associated and co-occurring features", 9),
    rep("Report elements and other", 3)
  ),
  Domain = c(
    "Social-emotional reciprocity",
    "Nonverbal communication",
    "Social relationships",
    "Stereotyped behavior",
    "Insistence on sameness",
    "Restricted interests",
    "Sensory reactivity",
    "Externalizing behaviors",
    "Internalizing behaviors",
    "Language skills",
    "Physiological function",
    "Adaptive behavior",
    "Intellectual functioning and learning skills",
    "Executive function",
    "Motor skills",
    "Family environment",
    "Test scores",
    "Other/general",
    "Recommendations"
  ),
  Definition = c(
    "Eye contact, response to name, joint attention, deficits in back-and-forth conversation, social and responsive smiling.",
    "Use of gestures, facial expression, and body language, and the use and understanding of nonverbal communication.",
    "Forming peer relationships, adjusting behavior to context, deficits in imaginative play, social and imitative play.",
    "Hand flapping, body rocking, lining up objects, echolalia, repetitive movements.",
    "Resistance to change in routines, ritualized communication patterns, insistence on routine, difficulty with transitions.",
    "Abnormally intense preoccupation with specific topics, restricted and fixed interests.",
    "Over-responsiveness to sound, fixation on textures, visual seeking, sensory hyper- or hypo-reactivity.",
    "Aggression, self-injury, disruptive behavior, anger outbursts, with meltdown distinguished from shutdown.",
    "Anxiety, depression, withdrawal, fear.",
    "Functional language ability, receptive and expressive level, everyday language use, sentence complexity, currently observed language function.",
    "Sleep patterns, eating habits including food selectivity, toileting.",
    "Daily-living and self-care skills, social independence.",
    "Verbal and nonverbal intelligence test results, learning ability.",
    "Planning, working memory, inhibitory control.",
    "Gross and fine motor development, coordination.",
    "Parenting stress, family support system, medical and developmental history.",
    "Scale scores such as ADOS, CARS, and K-WISC, and quantitative descriptions.",
    "General interview content not directly tied to symptoms, background information, demographics, referral reason.",
    "Recommendations for education or support based on the assessment results."
  ),
  check.names = FALSE,
  stringsAsFactors = FALSE
)

write_table(df, "Table3")
