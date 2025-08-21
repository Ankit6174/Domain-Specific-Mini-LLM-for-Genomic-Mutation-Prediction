gsap.to(".date-container, .day-month-container, .form-top-container-line, .form-home-container-button", {
  x: 0,
  opacity: 1,
  duration: 1,
  stagger: 0.1,
  ease: "power4.out",
  delay: 0.2,
});

gsap.to("form textarea, form button", {
  y: 0,
  opacity: 1,
  duration: 0.5,
  stagger: 0.1,
  ease: "power4.out",
  delay: 0.2,
});