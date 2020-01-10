using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Web.Http;
using System.Web.Http.Description;
using WebApplication1.Models;

namespace WebApplication1.Controllers
{
    public class CarsController : ApiController
    {
        private Car[] Cars = new Car[]
        {
        new Car { id = 1, leftspeed = 0, rightspeed= 0, leftlinesensor= false, rightlinesensor=false,ultrasonicsensor=0},
        new Car { id = 2, leftspeed = 0, rightspeed= 0, leftlinesensor= false, rightlinesensor=false,ultrasonicsensor=0},
        new Car { id = 3, leftspeed = 0, rightspeed= 0, leftlinesensor= false, rightlinesensor=false,ultrasonicsensor=0},
        new Car { id = 4, leftspeed = 0, rightspeed= 0, leftlinesensor= false, rightlinesensor=false,ultrasonicsensor=0},
        new Car { id = 5, leftspeed = 0, rightspeed= 0, leftlinesensor= false, rightlinesensor=false,ultrasonicsensor=0}
        };

        // GET: api/Cars
        [ResponseType(typeof(IEnumerable<Car>))]
        public IEnumerable<Car> Get()
        {
            return Cars;
        }

        // GET: api/Cars/5
        public IHttpActionResult Get(int id)
        {
            var product = Cars.FirstOrDefault((p) => p.id == id);
            if (product == null)
            {
                return NotFound();
            }
            return Ok(product);
        }

        // POST: api/Cars
        public void Post([FromBody]string value)
        {
            
        }

        // PUT: api/Cars/5
        public IHttpActionResult Put(int id, [FromBody]Car value)
        {
            
            var product = Cars.FirstOrDefault((p) => p.id == value.id);
            if (product == null)
            {
                return NotFound();
            }
             
            Cars[product.id - 1] = new Car { id = product.id, leftspeed = product.leftspeed,rightspeed= product.rightspeed, leftlinesensor = product.leftlinesensor, rightlinesensor = product.rightlinesensor, ultrasonicsensor= product.ultrasonicsensor };
            return Ok();
        }

        // DELETE: api/Cars/5
        public void Delete(int id)
        {
        }
    }
}
