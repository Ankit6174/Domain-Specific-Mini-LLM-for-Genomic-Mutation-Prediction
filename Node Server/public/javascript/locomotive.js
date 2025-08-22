gsap.registerPlugin(ScrollTrigger);

const scrollContainer = document.querySelector("#main");

let locoScroll;
const isMobile = window.innerWidth <= 300;

if (!isMobile) {
  locoScroll = new LocomotiveScroll({
    el: scrollContainer,
    smooth: true,
  });

  locoScroll.on("scroll", ScrollTrigger.update);

  ScrollTrigger.scrollerProxy(scrollContainer, {
    scrollTop(value) {
      return arguments.length
        ? locoScroll.scrollTo(value, 0, 0)
        : locoScroll.scroll.instance.scroll.y;
    },
    getBoundingClientRect() {
      return {
        top: 0,
        left: 0,
        width: window.innerWidth,
        height: window.innerHeight,
      };
    },
    pinType: scrollContainer.style.transform ? "transform" : "fixed",
  });

  ScrollTrigger.addEventListener("refresh", () => locoScroll.update());
  ScrollTrigger.refresh();
} else {
  ScrollTrigger.scrollerProxy(scrollContainer, {
    scrollTop(value) {
      return arguments.length ? window.scrollTo(0, value) : window.pageYOffset;
    },
    getBoundingClientRect() {
      return {
        top: 0,
        left: 0,
        width: window.innerWidth,
        height: window.innerHeight,
      };
    },
    pinType: "fixed",
  });

  ScrollTrigger.refresh();
}

document.querySelectorAll(".nav-link").forEach((link) => {
  link.addEventListener("click", (e) => {
    e.preventDefault();
    const targetId = link.getAttribute("href").substring(1);
    const targetElem = document.getElementById(targetId);

    if (!targetElem) return;

    if (!isMobile && locoScroll) {
      locoScroll.scrollTo(targetElem);
    } else {
      window.scrollTo({
        top: targetElem.offsetTop,
        behavior: "smooth",
      });
    }
  });
});
