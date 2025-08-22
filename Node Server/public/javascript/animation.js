gsap.to(".navbar-ion-container a", {
  x: 0,
  opacity: 1,
  duration: 1,
  stagger: 0.1,
  ease: "power4.out",
  delay: 0.5,
});

gsap.to(".navbar-predict-container a", {
  x: 0,
  opacity: 1,
  duration: 0.2,
  ease: "power4.out",
  delay: 0.7,
});

gsap.to(".scroll-animation-container", {
  y: 0,
  opacity: 1,
  duration: 0.7,
  ease: "power4.out",
  delay: 0.7,
});

gsap.to(".hero-section-main-heading span", {
  y: 0,
  opacity: 1,
  duration: 1,
  stagger: 0.1,
  ease: "power4.out",
  delay: 1,
});

// gsap.to(".hero-button-container", {
//   x: 0,
//   opacity: 1,
//   duration: 1,
//   ease: "power4.out",
//   delay: 1.5,
// });

// gsap.to(".hero-section-bottom-container p", {
//   x: 0,
//   opacity: 1,
//   duration: 1,
//   ease: "power4.out",
//   delay: 1.5,
// });

gsap.to(".hero-button-container", {
  x: 0,
  opacity: 1,
  duration: 1,
  ease: "power4.out",
  delay: 1.5,
  scrollTrigger: {
    trigger: ".hero-button-container",
    scroller: "#main",
    start: "top 90%",
  },
});

gsap.to(".hero-section-bottom-container p", {
  x: 0,
  opacity: 1,
  duration: 1,
  ease: "power4.out",
  delay: 1.5,
  scrollTrigger: {
    trigger: ".hero-section-bottom-container p",
    scroller: "#main",
    start: "top 90%",
  },
});

gsap.registerPlugin(ScrollTrigger);

gsap.to(
  ".call-to-action h1, .button-contianer-call-to-action, .footer-icon-cto",
  {
    y: 0,
    opacity: 1,
    duration: 1.2,
    ease: "power4.out",
    stagger: 0.1,
    scrollTrigger: {
      trigger:
        ".call-to-action h1, .button-contianer-call-to-action, .footer-icon-cto",
      start: "top 80%",
      scroller: "#main",
    },
  }
);

gsap.to(
  ".about-container-main-heading, .about-main-content-main-top-container, .about-main-content-top-container-paragraph, .about-main-content-bottom-container p, .about-main-content-bottom-container-icon-container",
  {
    y: 0,
    opacity: 1,
    duration: 2,
    ease: "power4.out",
    stagger: 0.2,
    scrollTrigger: {
      trigger:
        ".about-container-main-heading, .about-main-content-main-top-container, .about-main-content-top-container-paragraph, .about-main-content-bottom-container p, .about-main-content-bottom-container-icon-container",
      start: "top 80%",
      scroller: "#main",
    },
  }
);

gsap.to(".about-right-center-container-items", {
  y: 0,
  opacity: 1,
  duration: 1.5,
  delay: 0.4,
  ease: "power4.out",
  stagger: 0.2,
  scrollTrigger: {
    trigger: ".about-right-center-container-items",
    start: "top 80%",
    scroller: "#main",
  },
});

gsap.to(".about-box-containerr", {
  y: 0,
  opacity: 1,
  duration: 1.5,
  delay: 0.2,
  ease: "power4.out",
  stagger: 0.2,
  scrollTrigger: {
    trigger: ".about-box-containerr",
    start: "top 80%",
    scroller: "#main",
  },
});

gsap.to(".about-box-container", {
  y: 0,
  opacity: 1,
  duration: 1.5,
  delay: 0.2,
  ease: "power4.out",
  stagger: 0.2,
  scrollTrigger: {
    trigger: ".about-box-container",
    start: "top 80%",
    scroller: "#main",
  },
});

gsap.to(".contact-main-heading, .contact-main-paragraph", {
  y: 0,
  opacity: 1,
  duration: 1,
  delay: 0.4,
  ease: "power4.out",
  stagger: 0.2,
  scrollTrigger: {
    trigger: ".contact-main-heading, .contact-main-paragraph",
    start: "top 80%",
    scroller: "#main",
  },
});

gsap.to(
  ".name-email-input-contact-container, .organization-input-field-contact-container, .what-in-you-mind-input-field-contact-container, .message-input-field-contact-container, .contact-page-main-send-button",
  {
    y: 0,
    opacity: 1,
    duration: 2,
    ease: "power4.out",
    stagger: 0.2,
    scrollTrigger: {
      trigger:
        ".name-email-input-contact-container, .organization-input-field-contact-container, .what-in-you-mind-input-field-contact-container, .message-input-field-contact-container, .contact-page-main-send-button",
      start: "top 80%",
      scroller: "#main",
    },
  }
);

gsap.to(
  ".contact-page-visit-us-container h2, .contact-page-visit-us-container p, .contact-page-talk-us-to-container h2, .contact-page-talk-us-to-container p",
  {
    y: 0,
    opacity: 1,
    duration: 1,
    delay: 0.5,
    ease: "power4.out",
    stagger: 0.2,
    scrollTrigger: {
      trigger:
        ".contact-page-visit-us-container h2, .contact-page-visit-us-container p, .contact-page-talk-us-to-container h2, .contact-page-talk-us-to-container p",
      start: "top 80%",
      scroller: "#main",
    },
  }
);

gsap.to(".contact-page-icons-container i", {
  y: 0,
  opacity: 1,
  duration: 1,
  delay: 0.2,
  ease: "power4.out",
  stagger: 0.1,
  scrollTrigger: {
    trigger: ".contact-page-icons-container i",
    start: "top 80%",
    scroller: "#main",
  },
});
