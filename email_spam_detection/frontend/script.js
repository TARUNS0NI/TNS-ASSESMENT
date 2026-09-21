// ==================================================================
// FRONTEND LOGIC: script.js
//
// Sends the message text to the FastAPI backend and renders the verdict.
// The backend must be running (python app.py) before this page can work.
// ==================================================================


// ------------------------------------------------------------------
// STEP 1: CONFIGURATION
// ------------------------------------------------------------------
// Address of the /predict endpoint exposed by app.py on port 8000
const API_URL = "http://127.0.0.1:8000/predict";


// ------------------------------------------------------------------
// STEP 2: GRAB THE ELEMENTS WE NEED FROM THE PAGE
// ------------------------------------------------------------------
const emailText    = document.getElementById("emailText");
const checkBtn     = document.getElementById("checkBtn");
const resultBox    = document.getElementById("result");
const stamp        = document.getElementById("stamp");
const stampVerdict = document.getElementById("stampVerdict");
const readoutLine  = document.getElementById("readoutLine");
const readoutMeta  = document.getElementById("readoutMeta");
const statusLine   = document.getElementById("status");


// ------------------------------------------------------------------
// STEP 3: SAMPLE MESSAGES FOR THE QUICK-FILL CHIPS
// ------------------------------------------------------------------
const SAMPLES = {
  spam: "URGENT: You have been selected to receive a $1000 cash bonus. Click this link now to claim your free prize before the offer expires!",
  ham:  "Hi, just confirming our project review meeting has moved to Thursday at 3pm. I have attached the updated agenda, let me know if that time still works for you."
};

// Fill the textarea when a chip is clicked, and clear any previous verdict
document.querySelectorAll(".chip").forEach(function (chip) {
  chip.addEventListener("click", function () {
    emailText.value = SAMPLES[chip.dataset.sample];
    clearResult();
    emailText.focus();
  });
});


// ------------------------------------------------------------------
// STEP 4: HELPERS
// ------------------------------------------------------------------

// Hide the verdict area, used before every new request
function clearResult() {
  resultBox.classList.remove("visible");
  statusLine.textContent = "";
  statusLine.classList.remove("error");
}

// Show a message in the status line, optionally styled as an error
function setStatus(message, isError) {
  statusLine.textContent = message;
  statusLine.classList.toggle("error", Boolean(isError));
}


// ------------------------------------------------------------------
// STEP 5: SEND THE TEXT TO THE MODEL AND RENDER THE RESULT
// ------------------------------------------------------------------
async function checkMessage() {
  const text = emailText.value.trim();

  // Guard against an empty submission before troubling the server
  if (!text) {
    clearResult();
    setStatus("Enter some message text first.", true);
    emailText.focus();
    return;
  }

  // Put the button into its working state so the click clearly registered
  clearResult();
  checkBtn.disabled = true;
  checkBtn.textContent = "Checking…";

  try {
    // POST the message as JSON, matching the EmailInput schema in app.py
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email_text: text })
    });

    // Any non-200 status means the API rejected the request or failed internally
    if (!response.ok) {
      const detail = await response.text();
      setStatus("The server returned an error (" + response.status + "): " + detail, true);
      return;
    }

    // Parse the JSON body the backend sent back
    const result = await response.json();
    render(result);

  } catch (error) {
    // fetch() throws here when the backend is not running or is unreachable
    setStatus(
      "Cannot reach the API. Start the backend with 'python app.py' and make sure it is on port 8000.",
      true
    );
  } finally {
    // Always restore the button, whether the request succeeded or failed
    checkBtn.disabled = false;
    checkBtn.textContent = "Check this message";
  }
}


// ------------------------------------------------------------------
// STEP 6: PAINT THE VERDICT ONTO THE PAGE
// ------------------------------------------------------------------
function render(result) {
  const isSpam = result.prediction === 1;

  // Colour the rubber stamp red for spam, blue for a message that passed
  stamp.classList.remove("spam", "safe");
  stamp.classList.add(isSpam ? "spam" : "safe");
  stampVerdict.textContent = isSpam ? "Spam" : "Clear";

  // Plain-language summary beside the stamp
  readoutLine.textContent = isSpam
    ? "This message looks like spam."
    : "This message looks legitimate.";

  // The numbers behind the decision
  readoutMeta.textContent =
    "confidence " + result.confidence + "%  ·  spam probability " + result.spam_probability + "%";

  // Revealing the section also triggers the stamp-press animation
  resultBox.classList.add("visible");
}


// ------------------------------------------------------------------
// STEP 7: WIRE UP THE EVENTS
// ------------------------------------------------------------------
checkBtn.addEventListener("click", checkMessage);

// Ctrl+Enter (or Cmd+Enter) submits, a familiar shortcut inside a text box
emailText.addEventListener("keydown", function (event) {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    checkMessage();
  }
});
