function selectTopic(topic) {
  document.getElementById("mindTopic").value = topic;

  let buttons = document.querySelectorAll("#topics button");
  buttons.forEach((btn) => btn.classList.remove("selected"));

  event.target.classList.add("selected");
  console.log(document.getElementById("mindTopic").value);
}
