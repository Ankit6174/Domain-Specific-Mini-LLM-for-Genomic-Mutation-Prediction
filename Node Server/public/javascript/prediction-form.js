const submitBtn = document.getElementById("btn2");
const dnaInput = document.getElementById("dnasequence");
const refInput = document.getElementById("reference");
const altInput = document.getElementById("alternate");
const mutationInput = document.getElementById("mutation");
const chromosomeInput = document.getElementById("chromosome");
const genomicPosInput = document.getElementById("genomicPosition");

submitBtn.disabled = true;

function validateForm() {
  const dnaValue = dnaInput.value;
  const refValue = refInput.value.toUpperCase();
  const altValue = altInput.value.toUpperCase();
  const mutationValue = mutationInput.value;
  const chromosomeValue = chromosomeInput.value;
  const genomicPosValue = genomicPosInput.value;

  // Rule 1: DNA sequence length must be greater than 200
  const isDnaValid = dnaValue.length > 200;

  // Rule 2: Reference and Alternate must be a single, valid nucleotide (A, T, G, C)
  const validNucleotide = /^[ATGC]$/;
  const isRefValid = refValue.length === 1 && validNucleotide.test(refValue);
  const isAltValid = altValue.length === 1 && validNucleotide.test(altValue);

  // Rule 3: Other fields must not be empty
  const isMutationValid = mutationValue !== "";
  const isChromosomeValid = chromosomeValue !== "";
  const isGenomicPosValid = genomicPosValue.trim() !== "";

  // The button is disabled if any of the validation rules fail
  submitBtn.disabled = !(isDnaValid && isRefValid && isAltValid && isMutationValid && isChromosomeValid && isGenomicPosValid);
}

// Add event listeners to all relevant form fields to validate on any change
[dnaInput, refInput, altInput, mutationInput, chromosomeInput, genomicPosInput].forEach(field => {
  const eventType = field.tagName.toLowerCase() === 'select' ? 'change' : 'input';
  field.addEventListener(eventType, validateForm);
});