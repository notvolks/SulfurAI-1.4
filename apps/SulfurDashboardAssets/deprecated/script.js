document.addEventListener("mousemove", (event) => {
  let glint = document.querySelector(".glint-tracking");
  if (!glint) {
    glint = document.createElement("div");
    glint.className = "glint-tracking";
    document.body.appendChild(glint);
  }
  glint.style.left = event.clientX + "px";
  glint.style.top = event.clientY + "px";
});
