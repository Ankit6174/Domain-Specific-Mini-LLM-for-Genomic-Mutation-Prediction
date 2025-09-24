function printReport() {
    // Hide the dashboard content
    document.getElementsByClassName('report-container')[0].style.display = 'block';
    for (let cards of ['date-day-container', 'second-row', 'third-row', 'forth-row', 'sixth-row']) {
      const element = document.getElementsByClassName(cards)[0];
      if (element) {
        element.style.display = 'none';
      }
    };
    document.body.style.padding = '0px';

    // Print the page
    window.print();

    setTimeout(() => {
        document.getElementsByClassName('report-container')[0].style.display = 'none';
        for (let cards of ['date-day-container', 'second-row', 'third-row', 'forth-row', 'sixth-row']) {
        const element = document.getElementsByClassName(cards)[0];
        if (element) {
            element.style.display = 'flex';
        }
        };
        document.body.style.padding = '20px 30px';
    }, 3000);
}

const today = new Date();

const date = today.getDate(); 
const day = today.toLocaleString('default', { weekday: 'long' }); 
const month = today.toLocaleString('default', { month: 'long' }); 

function getShortDayName(dayName) {
    if (!dayName || typeof dayName !== 'string') return null;

    const shortForms = {
        monday: 'Mon',
        tuesday: 'Tue',
        wednesday: 'Wed',
        thursday: 'Thu',
        friday: 'Fri',
        saturday: 'Sat',
        sunday: 'Sun'
    };

    return shortForms[dayName.toLowerCase()] || null;
}

document.getElementById('date').textContent = date;
document.getElementById('day').textContent = getShortDayName(day);
document.getElementById('month').textContent = month;