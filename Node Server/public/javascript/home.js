function selectTopic(topic) {
  document.getElementById("mindTopic").value = topic;

  let buttons = document.querySelectorAll("#topics button");
  buttons.forEach((btn) => btn.classList.remove("selected"));

  event.target.classList.add("selected");
  console.log(document.getElementById("mindTopic").value);
}

const form = document.getElementById("contectFrom");
const submitButton = document.getElementById("submitBtn");
const message = document.getElementById("successMessage");

form.addEventListener("submit", function(e) {
  e.preventDefault();

  submitButton.disabled = true;

  const formData = new FormData(form);

  fetch('/postContect', {
    method: "POST",
    body: new URLSearchParams(formData)
  })
  .then(res => {
    if (res.ok) {
      alert("Form data sent successfully");
      message.style.display = 'flex';
      form.reset();
    } else {
      alert("Something Went Wrong");
    }
  })
  .catch((err) => {
    alert("Error while submitting form!");
    console.log(err);
  })
  .finally(() => {
    submitButton.disabled = false;
  });
});