export const QUIZ_SECTIONS = [
  {
    id: 'schedule',
    title: 'Schedule',
    icon: 'calendar'
  },
  {
    id: 'activity',
    title: 'Activity Level',
    icon: 'activity'
  },
  {
    id: 'interests',
    title: 'Interests',
    icon: 'heart'
  }
];

export const QUIZ_QUESTIONS = [
  {
    id: 1,
    sectionId: 'schedule',
    question: "What is your typical daily schedule?",
    type: "multiple-choice",
    options: [
      "Early mornings (5-8 AM)",
      "Mid-day (10 AM - 2 PM)",
      "Evenings (5-8 PM)",
      "Flexible throughout the day"
    ]
  },
  {
    id: 2,
    sectionId: 'activity',
    question: "How active are you with your pet?",
    type: "multiple-choice",
    options: [
      "Very active - multiple long sessions daily",
      "Moderately active - regular walks and playtime",
      "Lightly active - short walks and indoor play",
      "Varies based on mood and weather"
    ]
  },
  {
    id: 3,
    sectionId: 'interests',
    question: "What activities do you enjoy?",
    type: "multi-select",
    options: [
      "Hiking and outdoor adventures",
      "Dog parks and socialization",
      "Training and obedience",
      "Beach or water activities"
    ]
  }
];

// Keep backward compatibility
export const quizQuestions = QUIZ_QUESTIONS;
